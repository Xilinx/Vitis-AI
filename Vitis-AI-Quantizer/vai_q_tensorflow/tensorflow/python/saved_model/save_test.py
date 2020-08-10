# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for trackable object SavedModel save."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import compat


class _ModelWithOptimizer(util.Checkpoint):

  def __init__(self):
    self.dense = core.Dense(1)
    self.optimizer = adam.Adam(0.01)

  @def_function.function(
      input_signature=(tensor_spec.TensorSpec([None, 2], dtypes.float32),
                       tensor_spec.TensorSpec([None], dtypes.float32)))
  def call(self, x, y):
    with backprop.GradientTape() as tape:
      loss = math_ops.reduce_mean((self.dense(x) - y) ** 2.)
    trainable_variables = self.dense.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    return {"loss": loss}


def _import_and_infer(
    save_dir, inputs,
    signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
  """Import a SavedModel into a TF 1.x-style graph and run `signature_key`."""
  graph = ops.Graph()
  with graph.as_default(), session_lib.Session() as session:
    model = loader.load(session, [tag_constants.SERVING], save_dir)
    signature = model.signature_def[signature_key]
    assert set(inputs.keys()) == set(signature.inputs.keys())
    feed_dict = {}
    for arg_name in inputs.keys():
      feed_dict[graph.get_tensor_by_name(signature.inputs[arg_name].name)] = (
          inputs[arg_name])
    output_dict = {}
    for output_name, output_tensor_info in signature.outputs.items():
      output_dict[output_name] = graph.get_tensor_by_name(
          output_tensor_info.name)
    return session.run(output_dict, feed_dict=feed_dict)


class SaveTest(test.TestCase):

  def test_method_save_signature(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir, root.f)
    self.assertEqual(
        {"output_0": 2.},
        _import_and_infer(save_dir, {"x": 1.}))

  def test_method_save_concrete(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda z: {"out": 2. * z})
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(
        root,
        save_dir,
        {"non_default_key": root.f.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32))})
    self.assertEqual(
        {"out": 2.},
        _import_and_infer(
            save_dir, {"z": 1.}, signature_key="non_default_key"))

  def test_unbuilt_model_does_not_prevent_saving(self):
    root = util.Checkpoint(model=sequential.Sequential([core.Dense(2)]))
    save.save(root, os.path.join(self.get_temp_dir(), "saved_model"))

  def test_captured_symbolic_tensor_exception(self):
    root = module.Module()
    symbolic_tensor = []

    @def_function.function
    def captured_intermediate(x):
      symbolic_tensor.append(math_ops.add(x, x, name="a_tensor"))
      return symbolic_tensor[-1] * 2

    captured_intermediate(constant_op.constant(1.))

    root.f = def_function.function(lambda: symbolic_tensor[-1],
                                   input_signature=[])
    with self.assertRaisesRegexp(ValueError, "a_tensor"):
      save.save(root, os.path.join(self.get_temp_dir(), "saved_model"),
                signatures=root.f)

  def test_unsaveable_func_graph(self):
    root = module.Module()

    @def_function.function(input_signature=[])
    def nested_f():
      ops.get_default_graph().mark_as_unsaveable("ERROR MSG")
      return 1

    @def_function.function(input_signature=[])
    def f():
      return nested_f()

    root.f = f
    with self.assertRaisesRegexp(ValueError, "ERROR MSG"):
      save.save(root, os.path.join(self.get_temp_dir(), "saved_model"))

  def test_version_information_included(self):
    root = tracking.AutoTrackable()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    saved_model_proto = loader_impl.parse_saved_model(save_dir)
    self.assertEqual(
        versions.__version__,
        saved_model_proto.meta_graphs[0].meta_info_def.tensorflow_version)
    self.assertEqual(
        versions.__git_version__,
        saved_model_proto.meta_graphs[0].meta_info_def.tensorflow_git_version)

  def test_non_concrete_error(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
    root.f(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(
        ValueError, "Expected a TensorFlow function"):
      save.save(root, save_dir, root.f)

  def test_captures_unreachable_variable(self):
    root = tracking.AutoTrackable()
    unreachable_variable = variables.Variable([5.0, 2.0])
    root.reachable_variable = variables.Variable([1.0, 3.0])

    @def_function.function
    def increase_variable(x):
      return 2 * unreachable_variable * x + root.reachable_variable

    root.f = increase_variable

    self.assertAllEqual([101.0, 83.0],
                        root.f(constant_op.constant([10.0, 20.0])).numpy())

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")

    with self.assertRaisesRegexp(KeyError, "not reachable from root"):
      save.save(root, save_dir)

  def test_nested_inputs(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x[0],
        input_signature=([tensor_spec.TensorSpec(None, dtypes.float32),
                          tensor_spec.TensorSpec(None, dtypes.float32)],))
    root.f([constant_op.constant(1.), constant_op.constant(1.)])

  def test_nested_outputs(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: (2. * x, (3. * x, 4. * x)))
    root.f(constant_op.constant(1.))
    to_save = root.f.get_concrete_function(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(
        ValueError, "non-flat outputs"):
      save.save(root, save_dir, to_save)

  def test_nested_dict_outputs(self):
    root = util.Checkpoint(
        f=def_function.function(
            lambda x: {"a": 2. * x, "b": (3. * x, 4. * x)}))
    root.f(constant_op.constant(1.))
    to_save = root.f.get_concrete_function(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(
        ValueError, "dictionary containing non-Tensor value"):
      save.save(root, save_dir, to_save)

  def test_variable(self):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(
        lambda x: root.v1 * root.v2 * x)
    root.f(constant_op.constant(1.))
    to_save = root.f.get_concrete_function(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir, to_save)
    self.assertAllEqual({"output_0": 12.},
                        _import_and_infer(save_dir, {"x": 2.}))

  def test_optimizer(self):
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    model = _ModelWithOptimizer()
    first_loss = model.call(x, y)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(model, save_dir, model.call)
    second_loss = model.call(x, y)
    self.assertNotEqual(first_loss, second_loss)
    self.assertAllClose(
        second_loss,
        _import_and_infer(save_dir, {"x": [[3., 4.]], "y": [2.]}))

  def test_single_method_default_signature(self):
    model = _ModelWithOptimizer()
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    model.call(x, y)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(model, save_dir)
    self.assertIn("loss",
                  _import_and_infer(save_dir,
                                    {"x": [[3., 4.]], "y": [2.]}))

  def test_single_function_default_signature(self):
    model = tracking.AutoTrackable()
    model.f = def_function.function(lambda: 3., input_signature=())
    model.f()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(model, save_dir)
    self.assertAllClose({"output_0": 3.},
                        _import_and_infer(save_dir, {}))

  def test_single_function_no_signature(self):
    model = tracking.AutoTrackable()
    model.f = def_function.function(lambda: 3.)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(model, save_dir)

  def test_find_default_save_function(self):

    class ObjWithDefaultSignature(util.Checkpoint):

      @def_function.function(input_signature=[tensor_spec.TensorSpec(
          shape=None, dtype=dtypes.float32)])
      def _default_save_signature(self, x):
        return x + x + 1

    obj = ObjWithDefaultSignature()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(obj, save_dir)
    self.assertAllClose(
        {"output_0": 7.}, _import_and_infer(save_dir, {"x": 3.}))

  def test_docstring(self):

    class Adder(module.Module):

      @def_function.function(input_signature=[tensor_spec.TensorSpec(
          shape=None, dtype=dtypes.float32)])
      def add(self, x):
        return x + x + 1.

    to_save = Adder()
    to_save.add(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(to_save, save_dir)
    self.assertAllClose({"output_0": 7.},
                        _import_and_infer(save_dir, {"x": 3.}))

  def test_datastructures(self):

    class HasDatastructures(util.Checkpoint):

      def __init__(self):
        self.a = [1.]
        self.a.append(variables.Variable(2.))
        self.b = {"a": variables.Variable(3.)}

      @def_function.function(input_signature=[tensor_spec.TensorSpec(
          shape=None, dtype=dtypes.float32)])
      def add(self, x):
        return x + math_ops.add_n(self.a) + self.b["a"]

    to_save = HasDatastructures()
    to_save.add(constant_op.constant(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(to_save, save_dir)
    self.assertAllClose({"output_0": 10.},
                        _import_and_infer(save_dir, {"x": 4.}))

  def test_default_attr_stripping(self):

    class Complex(util.Checkpoint):

      @def_function.function(input_signature=[])
      def __call__(self):
        return math_ops.complex(
            constant_op.constant(1.),
            constant_op.constant(2.),
            name="complex")

    to_save = Complex()
    to_save()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(to_save, save_dir)
    graph = ops.Graph()
    with graph.as_default(), self.session(graph) as session:
      loader.load(session, [tag_constants.SERVING], save_dir)
      func, = [f for name, f in graph._functions.items() if "call" in name]
      complex_node, = [
          node for node in func.definition.node_def if node.op == "Complex"]
      self.assertNotIn("T", complex_node.attr)
      self.assertNotIn("Tout", complex_node.attr)

  def test_signature_attribute_reserved(self):
    root = util.Checkpoint(signatures=variables.Variable(1.))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegexp(ValueError, "del obj.signatures"):
      save.save(root, save_dir)
    del root.signatures
    save.save(root, save_dir)

  def test_function_with_captured_dataset(self):
    if test_util.is_gpu_available():
      self.skipTest("Currently broken when a GPU is available.")

    class HasDataset(module.Module):

      def __init__(self):
        super(HasDataset, self).__init__()
        self.dataset = (
            dataset_ops.Dataset.range(5)
            .map(lambda x: x ** 2))

      @def_function.function
      def __call__(self, x):
        current_sum = array_ops.zeros([], dtype=dtypes.int64)
        for element in self.dataset:
          current_sum += x * element
        return current_sum

    root = HasDataset()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(
        root, save_dir,
        signatures=root.__call__.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.int64)))
    self.assertAllClose({"output_0": 3 * (1 + 4 + 9 + 16)},
                        _import_and_infer(save_dir, {"x": 3}))


class AssetTests(test.TestCase):

  def setUp(self):
    super(AssetTests, self).setUp()
    self._vocab_path = os.path.join(self.get_temp_dir(), "vocab.txt")
    with open(self._vocab_path, "w") as f:
      f.write("alpha\nbeta\ngamma\n")

  def test_asset_path_returned(self):
    root = tracking.AutoTrackable()
    root.path = tracking.TrackableAsset(self._vocab_path)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    root.get_asset = def_function.function(lambda: root.path.asset_path)
    save.save(root, save_dir, signatures=root.get_asset.get_concrete_function())
    second_dir = os.path.join(self.get_temp_dir(), "second_dir")
    file_io.rename(save_dir, second_dir)
    imported_path = _import_and_infer(second_dir, {})["output_0"]
    self.assertIn(compat.as_str_any(second_dir),
                  compat.as_str_any(imported_path))

  def test_table(self):
    initializer = lookup_ops.TextFileInitializer(
        self._vocab_path,
        key_dtype=dtypes.string,
        key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
        value_dtype=dtypes.int64,
        value_index=lookup_ops.TextFileIndex.LINE_NUMBER)
    root = util.Checkpoint(table=lookup_ops.HashTable(
        initializer, default_value=-1))
    root.table_user = def_function.function(
        root.table.lookup,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.string)])
    self.assertEqual(
        2,
        self.evaluate(root.table_user(constant_op.constant("gamma"))))
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    file_io.delete_file(self._vocab_path)
    self.assertAllClose(
        {"output_0": [2, 0]},
        _import_and_infer(save_dir, {"keys": ["gamma", "alpha"]}))
    second_dir = os.path.join(self.get_temp_dir(), "second_dir")
    # Asset paths should track the location the SavedModel is loaded from.
    file_io.rename(save_dir, second_dir)
    self.assertAllClose(
        {"output_0": [2, 1]},
        _import_and_infer(second_dir, {"keys": ["gamma", "beta"]}))

  def test_unused_asset(self):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.asset = tracking.TrackableAsset(self._vocab_path)

    export_dir = os.path.join(self.get_temp_dir(), "save_dir")
    save.save(root, export_dir)
    self.assertAllClose(
        {"output_0": [0.2]},
        _import_and_infer(export_dir, {"x": [0.1]}))

  def test_sensible_function_building_exception(self):
    root = util.Checkpoint(v=variables.Variable(2.))
    root.f = def_function.function(
        lambda x: 2. * root.v,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    export_dir = os.path.join(self.get_temp_dir(), "save_dir")
    @def_function.function
    def _calls_save():
      save.save(root, export_dir)
    with self.assertRaisesRegexp(AssertionError, "tf.function"):
      _calls_save()


class _ModelWithOptimizerUsingDefun(util.Checkpoint):

  def __init__(self):
    self.dense = core.Dense(1)
    self.optimizer = adam.Adam(0.01)

  # Using defun due to control flow v2 cycles, b/121159261. def_function uses
  # conds to gate variable initialization and so triggers cond reference cycles,
  # but the thing being wrapped here does not use cond itself.
  @function.defun(
      input_signature=(tensor_spec.TensorSpec([None, 2], dtypes.float32),
                       tensor_spec.TensorSpec([None], dtypes.float32)),
  )
  def call(self, x, y):
    with backprop.GradientTape() as tape:
      loss = math_ops.reduce_mean((self.dense(x) - y) ** 2.)
    trainable_variables = self.dense.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    return {"loss": loss}


class MemoryTests(test.TestCase):

  def setUp(self):
    self._model = _ModelWithOptimizerUsingDefun()

  @test_util.assert_no_garbage_created
  def test_no_reference_cycles(self):
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    self._model.call(x, y)
    if sys.version_info[0] < 3:
      # TODO(allenl): debug reference cycles in Python 2.x
      self.skipTest("This test only works in Python 3+. Reference cycles are "
                    "created in older Python versions.")
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(self._model, save_dir, self._model.call)


if __name__ == "__main__":
  test.main()
