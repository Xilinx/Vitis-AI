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
"""Tests for trackable object SavedModel loading."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os
import sys
import tempfile
import weakref

from absl.testing import parameterized

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.eager import wrap_function
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as framework_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import monitored_session
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import tf_inspect


def cycle(obj, cycles, signatures=None):
  to_save = obj
  # TODO(vbardiovsky): It would be nice if exported protos reached a fixed
  # point w.r.t. saving/restoring, ideally after 2nd saving.
  for _ in range(cycles):
    path = tempfile.mkdtemp(prefix=test.get_temp_dir())
    # If available, we'll run the save and restore preferring the GPU. This
    # just makes sure we aren't throwing errors and have enough
    # device("CPU") blocks to satisfy the placer.
    with test_util.use_gpu():
      save.save(to_save, path, signatures)
      loaded = load.load(path)
    to_save = loaded
  return loaded


@parameterized.named_parameters(
    dict(testcase_name="ReloadOnce", cycles=1),
    dict(testcase_name="ReloadTwice", cycles=2),
    dict(testcase_name="ReloadThrice", cycles=3))
class LoadTest(test.TestCase, parameterized.TestCase):

  def test_structure_import(self, cycles):
    root = tracking.AutoTrackable()
    root.dep_one = tracking.AutoTrackable()
    root.dep_two = tracking.AutoTrackable()
    root.dep_two.dep = tracking.AutoTrackable()
    root.dep_three = root.dep_two.dep
    imported = cycle(root, cycles)
    self.assertIs(imported.dep_three, imported.dep_two.dep)
    self.assertIsNot(imported.dep_one, imported.dep_two)

  def test_variables(self, cycles):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(1., trainable=True)
    root.v2 = variables.Variable(2., trainable=False)
    imported = cycle(root, cycles)
    self.assertEqual(imported.v1.numpy(), 1.0)
    self.assertTrue(imported.v1.trainable)
    self.assertEqual(imported.v2.numpy(), 2.0)
    self.assertFalse(imported.v2.trainable)

  def test_variables_name(self, cycles):
    root = tracking.AutoTrackable()
    # Test 2 variables with same name: should work as the checkpoint
    # is based on object name and not on variable name.
    root.v1 = variables.Variable(1., trainable=True, name="v1")
    root.v2 = variables.Variable(2., trainable=False, name="v1")
    imported = cycle(root, cycles)
    self.assertEqual(imported.v1.numpy(), 1.0)
    self.assertEqual(imported.v2.numpy(), 2.0)
    self.assertEqual(imported.v1.name, root.v1.name)
    self.assertEqual(imported.v2.name, root.v2.name)
    with variable_scope.variable_scope("foo"):
      imported = cycle(root, cycles)
      self.assertTrue(imported.v1.name.startswith("foo/"))
      self.assertTrue(imported.v2.name.startswith("foo/"))

  def test_partially_defined_variable_shape(self, cycles):

    class MakeVariable(module.Module):

      def __init__(self):
        self.v = None

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec([None], dtypes.int64)])
      def make_variable(self, initial_value):
        if self.v is None:
          self.v = variables.Variable(initial_value)

    m = MakeVariable()
    m.make_variable([1, 2, 3])
    m = cycle(m, cycles)
    m.v.assign([1, 2, 3, 4])
    self.assertEqual([None], tensor_shape.as_shape(m.v.shape).as_list())

  @test_util.run_in_graph_and_eager_modes
  def test_capture_variables(self, cycles):
    root = tracking.AutoTrackable()
    root.weights = variables.Variable(2.)
    self.evaluate(root.weights.initializer)
    root.f = def_function.function(
        lambda x: root.weights * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    for _ in range(cycles):
      imported = cycle(root, 1)
      self.evaluate(imported.weights.initializer)
    self.assertEqual(4., self.evaluate(imported.f(constant_op.constant(2.))))
    self.evaluate(imported.weights.assign(4.0))
    self.assertEqual(8., self.evaluate(imported.f(constant_op.constant(2.))))

  @test_util.run_in_graph_and_eager_modes
  def test_capture_constant(self, cycles):
    root = tracking.AutoTrackable()
    captured_constant = constant_op.constant(2.)
    root.f = def_function.function(
        lambda x: captured_constant * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    imported = cycle(root, cycles)
    self.assertEqual(4., self.evaluate(imported.f(constant_op.constant(2.))))

  def test_control_outputs(self, cycles):
    exported = tracking.AutoTrackable()
    exported.v = variables.Variable(1.)
    exported.f = def_function.function(
        lambda: exported.v.assign(2., name="should_be_control_output"))
    exported_graph = exported.f.get_concrete_function().graph
    self.assertIn(
        exported_graph.get_operation_by_name("should_be_control_output"),
        exported_graph.control_outputs)

    imported = cycle(exported, cycles)
    # Calling get_concrete_function wraps in a second call operation; we want to
    # inspect the original function body for the control output; digging into
    # graph.as_graph_def() and its FunctionDefLibrary is another option.
    imported_concrete, = imported.f.concrete_functions
    imported_graph = imported_concrete.graph
    self.assertIn(
        imported_graph.get_operation_by_name("should_be_control_output"),
        imported_graph.control_outputs)

  def _make_asset(self, contents):
    filename = tempfile.mktemp(prefix=self.get_temp_dir())
    with open(filename, "w") as f:
      f.write(contents)
    return filename

  @test_util.run_in_graph_and_eager_modes
  def test_assets(self, cycles):
    file1 = self._make_asset("contents 1")
    file2 = self._make_asset("contents 2")

    root = tracking.AutoTrackable()
    root.asset1 = tracking.TrackableAsset(file1)
    root.asset2 = tracking.TrackableAsset(file2)

    save_dir = os.path.join(self.get_temp_dir(), "save_dir")
    save.save(root, save_dir)

    file_io.delete_file(file1)
    file_io.delete_file(file2)
    load_dir = os.path.join(self.get_temp_dir(), "load_dir")
    file_io.rename(save_dir, load_dir)

    imported = load.load(load_dir)
    with open(self.evaluate(imported.asset1.asset_path), "r") as f:
      self.assertEqual("contents 1", f.read())
    with open(self.evaluate(imported.asset2.asset_path), "r") as f:
      self.assertEqual("contents 2", f.read())

  def test_cond_prune(self, cycles):
    x_in = []
    x_out = []

    def f(x, y):
      x_in.append(x)
      xx = cond_v2.cond_v2(
          math_ops.less(1, 2),
          lambda: x + 1,
          lambda: x + 2,
      )
      x_out.append(xx)
      return xx, 2 * y

    f_wrapped = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtypes.float32)] * 2)
    f_pruned = f_wrapped.prune(x_in[0], [x_out[0]])

    class Adder(module.Module):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32)])
      def add(self, x):
        return f_pruned(x)

    root = Adder()
    root.add(constant_op.constant(1.))
    root = cycle(root, cycles)
    root.add(constant_op.constant(1.))

  def test_capture_assets(self, cycles):
    root = tracking.AutoTrackable()
    root.vocab = tracking.TrackableAsset(self._make_asset("contents"))
    root.f = def_function.function(
        lambda: root.vocab.asset_path,
        input_signature=[])
    imported = cycle(root, cycles)
    original_output = root.f().numpy()
    imported_output = imported.f().numpy()
    self.assertNotEqual(original_output, imported_output)
    with open(imported_output, "r") as f:
      self.assertEqual("contents", f.read())

  def test_capture_assets_in_graph(self, cycles):
    root = tracking.AutoTrackable()
    root.vocab = tracking.TrackableAsset(self._make_asset("contents"))
    root.f = def_function.function(
        lambda: root.vocab.asset_path,
        input_signature=[])

    original_output = root.f().numpy()

    if cycles > 1:
      root = cycle(root, cycles - 1)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    with ops.Graph().as_default():
      imported = load.load(path)
      imported_tensor = imported.f()
      with monitored_session.MonitoredSession() as sess:
        imported_output = sess.run(imported_tensor)
        self.assertNotEqual(original_output, imported_output)
        with open(imported_output, "r") as f:
          self.assertEqual("contents", f.read())

  def test_dedup_assets(self, cycles):
    vocab = self._make_asset("contents")
    root = tracking.AutoTrackable()
    root.asset1 = tracking.TrackableAsset(vocab)
    root.asset2 = tracking.TrackableAsset(vocab)
    imported = cycle(root, cycles)
    self.assertEqual(imported.asset1.asset_path.numpy(),
                     imported.asset2.asset_path.numpy())

  def test_implicit_input_signature(self, cycles):
    @def_function.function
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func

    # Add two traces.
    root.f(constant_op.constant(1.))
    root.f(constant_op.constant(1))

    imported = cycle(root, cycles)

    self.assertEqual(4., imported.f(constant_op.constant(2.)).numpy())
    self.assertEqual(14, imported.f(constant_op.constant(7)).numpy())

  def test_explicit_input_signature(self, cycles):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func

    imported = cycle(root, cycles)
    self.assertEqual(4., imported.f(constant_op.constant(2.0)).numpy())

  def test_explicit_save_signature(self, cycles):
    @def_function.function
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func

    imported = cycle(
        root, cycles, {
            "f":
                root.f.get_concrete_function(
                    tensor_spec.TensorSpec(None, dtypes.float32))
        })
    self.assertEqual(4., imported.f(constant_op.constant(2.0)).numpy())

  def test_nested_functions(self, cycles):
    f = def_function.function(
        lambda x: x*2.0,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    g = def_function.function(
        lambda x: f(x) + 1.0,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    root = tracking.AutoTrackable()
    root.g = g
    imported = cycle(root, cycles)
    imported.g(constant_op.constant([1.0]))

  def test_function_with_default_bool_input(self, cycles):

    def func(x, training=False):
      if training:
        return 2 * x
      else:
        return 7

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(7, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())

    imported = cycle(root, cycles)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(7, imported.f(constant_op.constant(2)).numpy())

  def test_function_with_default_none_input(self, cycles):

    def func(x, dtype=None):
      if dtype:
        return array_ops.zeros(shape=x.shape, dtype=dtype)
      else:
        return array_ops.zeros(shape=x.shape, dtype=dtypes.float32)

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    self.assertAllEqual([0.0, 0.0, 0.0],
                        root.f(constant_op.constant([1, 2, 3])).numpy())
    self.assertAllEqual([0.0, 0.0, 0.0],
                        root.f(constant_op.constant([1.0, 2.0, 3.0])).numpy())
    self.assertAllEqual([0.0, 0.0, 0.0, 0.0],
                        root.f(constant_op.constant([1, 2, 3, 4])).numpy())
    self.assertAllEqual([0, 0, 0],
                        root.f(
                            constant_op.constant([1.0, 2.0, 3.0]),
                            dtype=dtypes.int32).numpy())

    concrete_functions = root.f._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access
    self.assertEqual(4, len(concrete_functions))

    imported = cycle(root, cycles)

    self.assertAllEqual([0.0, 0.0, 0.0],
                        imported.f(constant_op.constant([1, 2, 3]),
                                   None).numpy())
    self.assertAllEqual([0.0, 0.0, 0.0],
                        imported.f(constant_op.constant([1.0, 2.0,
                                                         3.0])).numpy())
    self.assertAllEqual([0.0, 0.0, 0.0, 0.0],
                        imported.f(constant_op.constant([1, 2, 3, 4])).numpy())
    self.assertAllEqual([0, 0, 0],
                        imported.f(
                            constant_op.constant([1.0, 2.0, 3.0]),
                            dtype=dtypes.int32).numpy())

  def test_function_no_return(self, cycles):

    class TrackableWithOneVariable(tracking.AutoTrackable):

      def __init__(self, initial_value=0.0):
        super(TrackableWithOneVariable, self).__init__()
        self.variable = variables.Variable(initial_value)

      @def_function.function
      def increase(self, by=1.0):
        self.variable.assign_add(by)

    obj = TrackableWithOneVariable(5.0)

    obj.increase(constant_op.constant(10.0))
    self.assertEqual(15.0, obj.variable.numpy())
    obj.increase()
    self.assertEqual(16.0, obj.variable.numpy())

    imported = cycle(obj, cycles)

    imported.increase(constant_op.constant(10.0))
    self.assertEqual(26.0, imported.variable.numpy())
    imported.increase(constant_op.constant(1.0))
    self.assertEqual(27.0, imported.variable.numpy())

  def test_structured_inputs(self, cycles):

    def func(x, training=True):
      # x is a nested structure, we care about one particular tensor.
      _, (a, b) = x
      if training:
        return 2 * a["a"] + b
      else:
        return 7

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    x = constant_op.constant(10)
    y = constant_op.constant(11)

    input1 = [6, ({"a": x}, y)]
    input2 = [7, ({"a": x}, y)]  # Not compatible with input1 signature.
    input3 = [6, ({"a": y}, x)]  # Compatible with input1 signature.

    # Note: by only calling f(input1) before serialization, only inputs with
    # matching signature will be valid on the loaded model.
    self.assertEqual(31, root.f(input1).numpy())

    imported = cycle(root, cycles)

    with self.assertRaisesRegexp(ValueError,
                                 "Could not find matching function to call"):
      imported.f(input2)

    self.assertEqual(31, imported.f(input1).numpy())
    self.assertEqual(32, imported.f(input3).numpy())

  def test_structured_output(self, cycles):

    # Use fields with non-alphabetical order
    named_tuple_type = collections.namedtuple("NamedTupleHello", ["b", "a"])

    def func(input1, input2):
      named_tuple = named_tuple_type(a=input1 + input2, b=input1 * input2)
      return [named_tuple, input2, {"x": 0.5}]

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    result = root.f(constant_op.constant(2), constant_op.constant(3))

    self.assertEqual(5, result[0].a.numpy())
    self.assertEqual(6, result[0].b.numpy())
    self.assertEqual(["b", "a"], list(result[0]._asdict().keys()))
    self.assertEqual(3, result[1].numpy())
    self.assertEqual(0.5, result[2]["x"].numpy())

    imported = cycle(root, cycles)

    result = imported.f(constant_op.constant(2), constant_op.constant(5))
    self.assertEqual(7, result[0].a.numpy())
    self.assertEqual(10, result[0].b.numpy())
    self.assertEqual(["b", "a"], list(result[0]._asdict().keys()))
    self.assertEqual(5, result[1].numpy())
    self.assertEqual(0.5, result[2]["x"].numpy())

  def test_optimizer(self, cycles):

    class _HasOptimizer(module.Module):

      def __init__(self):
        super(_HasOptimizer, self).__init__()
        self.layer = core.Dense(1)
        self.optimizer = adam.Adam(0.01)

      @def_function.function
      def __call__(self, x):
        return self.layer(x)

      @def_function.function
      def train(self, x, y):
        with backprop.GradientTape() as tape:
          predicted = self(x)
          loss = math_ops.reduce_sum(math_ops.abs(y - predicted))
        train_vars = self.layer.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

    root = _HasOptimizer()
    train_input = dict(x=constant_op.constant([[1.]]),
                       y=constant_op.constant([[2.]]))
    root.train(**train_input)
    imported = cycle(root, cycles)
    self.assertAllClose(root.optimizer.learning_rate.numpy(),
                        imported.optimizer.learning_rate.numpy())
    self.assertAllClose(root(constant_op.constant([[-0.5]])),
                        imported(constant_op.constant([[-0.5]])))
    root.train(**train_input)
    imported.train(**train_input)
    self.assertAllClose(root(constant_op.constant([[-0.5]])),
                        imported(constant_op.constant([[-0.5]])))

  def test_positional_arguments(self, cycles):
    def func(x, training=False, abc=7.1, defg=7.7):
      del abc
      if training:
        return 2 * x
      if defg == 7:
        return 6
      else:
        return 7

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(7, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())
    self.assertEqual(6, root.f(constant_op.constant(1), defg=7.0).numpy())

    imported = cycle(root, cycles)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(7, imported.f(constant_op.constant(2)).numpy())
    self.assertEqual(6, imported.f(constant_op.constant(1), defg=7.0).numpy())

  def test_additional_kwargs(self, cycles):
    def func(x, training=False, **options):
      del options
      if training:
        return 2 * x
      else:
        return 7

    root = tracking.AutoTrackable()
    root.f = def_function.function(func)

    x = constant_op.constant(10)
    self.assertEqual(7, root.f(x, learning_rate=0.5, epochs=3).numpy())

    imported = cycle(root, cycles)

    with self.assertRaisesRegexp(ValueError,
                                 "Could not find matching function to call.*"):
      imported.f(x, learning_rate=0.5, epochs=4)

    self.assertEqual(7, imported.f(x, learning_rate=0.5, epochs=3).numpy())

  def test_member_function(self, cycles):
    class TrackableWithMember(tracking.AutoTrackable):

      def __init__(self):
        super(TrackableWithMember, self).__init__()
        self._some_value = 20

      @def_function.function
      def f(self, x, training=False):
        if training:
          return 2 * x
        else:
          return 7 + self._some_value

    root = TrackableWithMember()

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(27, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())

    imported = cycle(root, cycles)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(27, imported.f(constant_op.constant(2)).numpy())

  def test_side_effect_listing(self, cycles):
    class M(tracking.AutoTrackable):

      def __init__(self):
        super(M, self).__init__()
        self.var = None

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
      def f(self, x):
        if self.var is None:
          self.var = variables.Variable(2.)
        return x * self.var

    m = M()
    cycle(m, cycles)
    self.assertEqual(4.0, m.f(constant_op.constant(2.0)).numpy())

  def test_basic_backprop(self, cycles):
    weight = variables.Variable(1., trainable=True)
    bias = variables.Variable(0., trainable=True)
    g = def_function.function(
        lambda x: x*weight + bias,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    root = tracking.AutoTrackable()
    root.weight = weight
    root.bias = bias
    root.g = g
    imported = cycle(root, cycles)
    with backprop.GradientTape() as t:
      x = constant_op.constant([3.5])
      loss = imported.g(x)
      grad = t.gradient(loss, [imported.weight, imported.bias])
      self.assertAllClose(grad, [3.5, 1.0])

  def test_nested_backprop(self, cycles):
    weight = variables.Variable(1., trainable=True)
    bias = variables.Variable(0., trainable=True)

    # Note: this function gets called from other function defs via a
    # "PartitionedCall" op node.
    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.float32),
        tensor_spec.TensorSpec(None, dtypes.float32)])
    def mul(x, y):
      return x * y

    # Note: this function gets called from other function defs via a
    # "StatefulPartitionedCall" op node.
    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.float32)])
    def f(x):
      return mul(weight.read_value(), x)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.float32)])
    def g(x):
      return f(x) + bias,

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.float32)])
    def h(x):
      return g(x) + bias,

    root = tracking.AutoTrackable()
    root.weight = weight
    root.bias = bias
    root.g = h

    imported = cycle(root, cycles)
    with backprop.GradientTape() as t:
      x = constant_op.constant([3.5])
      loss = imported.g(x)
    grad = t.gradient(loss, [imported.weight, imported.bias])
    self.assertAllClose(grad, [3.5, 2.0])

  def test_callable(self, cycles):
    class M1(tracking.AutoTrackable):

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
      def __call__(self, x):
        return x

    root = tracking.AutoTrackable()
    root.m1 = M1()
    root.m2 = tracking.AutoTrackable()
    root.m2.__call__ = def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])(
            lambda x: x*3.0)
    imported = cycle(root, cycles)
    x = constant_op.constant(1.0)

    self.assertTrue(callable(imported.m1))
    self.assertAllEqual(root.m1(x), imported.m1(x))

    # Note: `root.m2` was not callable since `__call__` attribute was set
    # into the instance and not on the class. But after a serialization cycle
    # that starts to work.
    self.assertTrue(callable(imported.m2))
    self.assertAllEqual(root.m2.__call__(x), imported.m2(x))

    # Verify that user objects without `__call__` attribute are not callable.
    self.assertFalse(callable(imported))

  def test_chain_callable(self, cycles):
    func = def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])(
            lambda x: x*3.0)
    root = tracking.AutoTrackable()
    root.__call__ = tracking.AutoTrackable()
    root.__call__.__call__ = tracking.AutoTrackable()
    root.__call__.__call__.__call__ = func

    imported = cycle(root, cycles)
    self.assertTrue(callable(imported))
    x = constant_op.constant(1.0)
    self.assertAllEqual(imported(x).numpy(), 3.0)

  def test_load_in_graph_mode(self, cycles):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(1., name="v_one", trainable=False)
    root.v2 = variables.Variable(2., name="v_two", trainable=True)
    root.f = def_function.function(
        lambda x: root.v2 * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    if cycles > 1:
      root = cycle(root, cycles - 1)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    with ops.Graph().as_default() as g:
      imported = load.load(path)
      var_v1 = imported.v1
      self.assertFalse(var_v1.trainable)
      var_v2 = imported.v2
      self.assertTrue(var_v2.trainable)
      output = imported.f(constant_op.constant(2.))
      with monitored_session.MonitoredSession() as sess:
        self.assertEqual(1.0, sess.run(var_v1))
        self.assertEqual(4.0, sess.run(output))
      self.assertCountEqual([var_v1, var_v2],
                            g.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
      # load() should not add to TRAINABLE_VARIABLES. Higher levels of model
      # building control retraining or frozen use of imported SavedModels.
      self.assertCountEqual([],
                            g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

  def test_load_in_func_graph(self, cycles):
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(1.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(
        lambda x: root.v2 * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    if cycles > 1:
      root = cycle(root, cycles - 1)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)

    closure = tracking.AutoTrackable()
    @def_function.function
    def func(x):
      if not hasattr(closure, "model"):
        closure.model = load.load(path)
      return closure.model.f(x)

    inputs = constant_op.constant(2.)
    self.assertEqual(4.0, func(inputs).numpy())

  def test_soft_matching(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func

    self.assertAllEqual([2], root.f(constant_op.constant([1])).numpy())
    self.assertAllEqual([2, 4], root.f(constant_op.constant([1, 2])).numpy())

    concrete_functions = root.f._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access
    self.assertEqual(1, len(concrete_functions))

    imported = cycle(root, cycles)

    with self.assertRaisesRegexp(ValueError, "Python inputs incompatible"):
      # We cannot call the function with a constant of shape ().
      imported.f(constant_op.constant(2)).numpy()

    # TODO(vbardiovsky): When classes are revived with input_signatures, we
    # should also check that the calls below are not generating any more
    # concrete functions.
    self.assertAllEqual([2, 4, 6, 8],
                        imported.f(constant_op.constant([1, 2, 3, 4])).numpy())
    self.assertAllEqual([2, 4, 6],
                        imported.f(constant_op.constant([1, 2, 3])).numpy())

  def test_get_concrete_function(self, cycles):

    @def_function.function
    def func(x, training=False):
      if training:
        return 2 * x
      else:
        return 3 * x

    func.get_concrete_function(
        tensor_spec.TensorSpec([None], dtypes.int32), True)
    func.get_concrete_function(tensor_spec.TensorSpec([None], dtypes.float32))

    root = tracking.AutoTrackable()
    root.f = func

    imported = cycle(root, cycles)

    concrete = imported.f.get_concrete_function(
        training=True, x=tensor_spec.TensorSpec([None], dtypes.int32))

    self.assertAllEqual([2, 4, 6, 8],
                        concrete(x=constant_op.constant([1, 2, 3, 4])).numpy())
    with self.assertRaisesRegexp(ValueError,
                                 "Could not find matching function to call"):
      imported.f.get_concrete_function(
          tensor_spec.TensorSpec([None], dtypes.int32))
    imported.f.get_concrete_function(
        tensor_spec.TensorSpec([None], dtypes.int32), True)

  def test_concrete_function(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function()

    self.assertAllEqual([2], root.f(constant_op.constant([1])).numpy())
    self.assertAllEqual([2, 4], root.f(constant_op.constant([1, 2])).numpy())

    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(root, cycles, signatures={})

    self.assertAllEqual([2, 4, 6, 8],
                        imported.f(constant_op.constant([1, 2, 3, 4])).numpy())
    self.assertAllEqual([2, 4, 6],
                        imported.f(constant_op.constant([1, 2, 3])).numpy())

  def test_concrete_function_captures(self, cycles):

    class Root(module.Module):

      def __init__(self):
        self.v = variables.Variable(1.)
        self.v1 = variables.Variable(1.)

      @def_function.function(
          input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
      def use_v(self, x):
        return self.v + self.v1 + 1.

    root = Root()
    self.assertIn(root.v.handle,
                  root.use_v.get_concrete_function().graph.external_captures)
    for _ in range(cycles):
      root = cycle(root, 1, signatures=root.use_v.get_concrete_function())
    func_captures = root.use_v.get_concrete_function().graph.external_captures
    self.assertLen(func_captures, 2)
    self.assertTrue(any(root.v.handle is t for t in func_captures))
    self.assertTrue(any(root.v1.handle is t for t in func_captures))
    signature_captures = root.signatures[
        "serving_default"].graph.external_captures
    self.assertLen(signature_captures, 2)
    self.assertTrue(any(root.v.handle is t for t in signature_captures))
    self.assertTrue(any(root.v1.handle is t for t in signature_captures))

  def test_concrete_function_arg_names(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function()

    self.assertAllEqual([2], root.f(constant_op.constant([1])).numpy())

    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(root, cycles, signatures={})

    self.assertAllEqual([2, 4, 6],
                        imported.f(x=constant_op.constant([1, 2, 3])).numpy())

  def test_concrete_function_no_signature(self, cycles):
    @def_function.function
    def func(x):
      return 2 * x

    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function(constant_op.constant([1]))
    self.assertAllEqual([4], root.f(constant_op.constant([2])).numpy())
    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(root, cycles, signatures={})
    self.assertAllEqual([6],
                        imported.f(constant_op.constant([3])).numpy())

  def test_concrete_function_backprop(self, cycles):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.float32)])
    def func(x):
      return x ** 2.
    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function()

    def _compute_gradient(function):
      with backprop.GradientTape() as tape:
        inp = constant_op.constant(1.)
        tape.watch(inp)
        output = function(inp)
      return tape.gradient(output, inp)

    self.assertEqual(2., _compute_gradient(root.f).numpy())
    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(root, cycles, signatures={})
    self.assertEqual(2., _compute_gradient(imported.f).numpy())

  def test_revived_concrete_function_kwargs(self, cycles):

    @def_function.function
    def func(x, y):
      return x * (y + 1.)
    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function(
        tensor_spec.TensorSpec([], dtypes.float32),
        tensor_spec.TensorSpec([], dtypes.float32))
    self.assertEqual(8., root.f(y=constant_op.constant(3.),
                                x=constant_op.constant(2.)).numpy())
    # TODO(andresp): Fix exporting of loaded concrete functions as signatures.
    imported = cycle(root, cycles, signatures={})
    self.assertEqual(8., imported.f(y=constant_op.constant(3.),
                                    x=constant_op.constant(2.)).numpy())

  def test_revived_concrete_function_tensorspec_kwargs(self, cycles):

    @def_function.function
    def func(*args):
      x, y = args
      return x * (y + 1.)
    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function(
        tensor_spec.TensorSpec([], dtypes.float32, name="x"),
        tensor_spec.TensorSpec([], dtypes.float32, name="y"))
    self.assertEqual(8., root.f(y=constant_op.constant(3.),
                                x=constant_op.constant(2.)).numpy())
    imported = cycle(root, cycles, signatures={})
    self.assertEqual(8., imported.f(y=constant_op.constant(3.),
                                    x=constant_op.constant(2.)).numpy())

  def test_concrete_function_variable_argument(self, cycles):
    # TODO(allenl): Fix variables in input signatures.
    self.skipTest("Need to fix encoding of variables in inputs signatures")
    capture = variables.Variable(0)

    @def_function.function
    def func(v):
      v.assign_add(1)
      capture.assign_sub(1)

    vsave = variables.Variable(1)
    root = tracking.AutoTrackable()
    root.f = func.get_concrete_function(vsave)
    root.capture = capture
    self.assertEqual(1, vsave.numpy())
    root.f(vsave)
    self.assertEqual(2, vsave.numpy())
    self.assertEqual(-1, capture.numpy())
    imported = cycle(root, cycles)

    vload = variables.Variable(1)
    imported.f(vload)
    self.assertEqual(2, vload.numpy())
    imported.f(v=vload)
    self.assertEqual(3, vload.numpy())
    self.assertEqual(-3, imported.capture.numpy())
    self.assertEqual(-1, capture.numpy())

  def test_function_and_component(self, cycles):

    @def_function.function
    def func(v):
      return v + 1

    root = tracking.AutoTrackable()
    root.func = func
    root.concrete_func = func.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.int32))
    one = constant_op.constant(1)
    self.assertEqual(2, root.func(one).numpy())
    self.assertEqual(2, root.concrete_func(one).numpy())
    imported = cycle(root, cycles)
    self.assertEqual(2, imported.func(one).numpy())
    self.assertEqual(2, imported.concrete_func(one).numpy())

  def test_dict(self, cycles):
    root = tracking.AutoTrackable()
    root.variables = dict(a=variables.Variable(1.))
    root.variables["b"] = variables.Variable(2.)
    root.variables["c"] = 1
    root.funcs = dict(
        a=def_function.function(lambda: constant_op.constant(100.)))
    root.funcs["conc"] = root.funcs["a"].get_concrete_function()
    imported = cycle(root, cycles)
    self.assertEqual(1., imported.variables["a"].numpy())
    self.assertEqual(2., imported.variables["b"].numpy())
    self.assertEqual(set(["a", "b"]), set(imported.variables.keys()))
    self.assertEqual(100., imported.funcs["a"]().numpy())
    self.assertEqual(100., imported.funcs["conc"]().numpy())

  def test_list(self, cycles):
    root = tracking.AutoTrackable()
    root.variables = [variables.Variable(1.)]
    root.variables.append(1)
    root.variables.append(variables.Variable(3.))
    imported = cycle(root, cycles)
    self.assertEqual(1., imported.variables[0].numpy())
    self.assertEqual(3., imported.variables[2].numpy())
    self.assertIs(None, imported.variables[1])
    self.assertEqual(3, len(imported.variables))

  def test_functions_list(self, cycles):
    root = tracking.AutoTrackable()
    v1 = variables.Variable(1.)
    root.losses = [def_function.function(lambda: math_ops.reduce_sum(v1 ** 2))]
    root.variables = [v1]

    @def_function.function
    def _v2_loss():
      if len(root.variables) == 1:
        v2 = variables.Variable(2.)
        root.variables.append(v2)
      return math_ops.reduce_sum(root.variables[1] ** 2)

    root.losses.append(_v2_loss)
    self.assertAllClose([1., 4.], [loss() for loss in root.losses])
    imported = cycle(root, cycles)
    self.assertAllClose([1., 4.], [loss() for loss in imported.losses])
    imported.variables[0].assign(3.)
    imported.variables[1].assign(4.)
    self.assertAllClose([9., 16.], [loss() for loss in imported.losses])

  def test_captured_constant(self, cycles):
    const = array_ops.zeros([100])
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda: const + 1.)
    root.g = def_function.function(lambda: const + 2.)
    self.assertAllClose(array_ops.ones([100]), root.f())
    self.assertAllClose(2. * array_ops.ones([100]), root.g())
    imported = cycle(root, cycles)
    self.assertAllClose(array_ops.ones([100]), imported.f())
    self.assertAllClose(2. * array_ops.ones([100]), imported.g())
    # TODO(b/123408994): Use the public get_concrete_function.
    f_concrete = imported.f._list_all_concrete_functions_for_serialization()[0]
    g_concrete = imported.g._list_all_concrete_functions_for_serialization()[0]
    self.assertLen(f_concrete.captured_inputs, 1)
    self.assertLen(g_concrete.captured_inputs, 1)
    # We should be using the same captured EagerTensor in both functions, not
    # duplicating the constant.
    self.assertIs(f_concrete.captured_inputs[0],
                  g_concrete.captured_inputs[0])

  def test_functions_accessed_once(self, cycles):

    class Exported(tracking.AutoTrackable):

      def __init__(self):
        self._counter = 0

      @property
      def make_func(self):
        @def_function.function
        def f():
          return constant_op.constant(self._counter)
        f.get_concrete_function()  # force a trace
        self._counter += 1
        return f

    exported = Exported()
    imported = cycle(exported, cycles)
    self.assertEqual(0, imported.make_func().numpy())
    self.assertEqual(1, exported.make_func().numpy())

  def test_overwritten_signatures_error(self, cycles):
    exported = tracking.AutoTrackable()
    exported.f = def_function.function(lambda: constant_op.constant(1.))
    imported = cycle(
        exported, cycles,
        signatures={"key": exported.f.get_concrete_function()})
    self.assertEqual(1., imported.signatures["key"]()["output_0"].numpy())
    imported.signatures = {"key1": imported.signatures["key"]}
    with self.assertRaisesRegexp(ValueError, "signatures"):
      save.save(imported, tempfile.mkdtemp(prefix=self.get_temp_dir()))

  def test_signature_loading(self, cycles):

    class Exported(tracking.AutoTrackable):

      def __init__(self):
        self.v = variables.Variable(3.)

      @def_function.function
      def do(self, x):
        return self.v * x

    exported = Exported()
    imported = cycle(
        exported,
        cycles=1,
        signatures=exported.do.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32)))
    for _ in range(cycles - 1):
      imported = cycle(imported, cycles=1, signatures=imported.signatures)
    self.assertEqual(["serving_default"], list(imported.signatures.keys()))
    imported_function = imported.signatures["serving_default"]
    two = constant_op.constant(2.)
    self.assertEqual(6., imported_function(x=two)["output_0"].numpy())
    imported.v.assign(4.)
    self.assertEqual(8., imported_function(x=two)["output_0"].numpy())
    self.assertEqual(8., imported_function(two)["output_0"].numpy())
    with self.assertRaises(TypeError):
      # The signatures mapping is immutable
      imported.signatures["random_key"] = 3

  def test_multiple_argument_signatures_no_positional(self, cycles):

    class Exported(tracking.AutoTrackable):

      @def_function.function
      def do(self, x, y):
        return x + y

    exported = Exported()
    imported = cycle(
        exported, cycles=1, signatures=exported.do.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32),
            tensor_spec.TensorSpec(None, dtypes.float32)))
    for _ in range(cycles - 1):
      imported = cycle(imported, cycles=1, signatures=imported.signatures)
    with self.assertRaises(TypeError):
      imported.signatures["serving_default"](
          constant_op.constant(1.),
          y=constant_op.constant(2.))
    self.assertEqual(
        {"output_0": 3.},
        self.evaluate(imported.signatures["serving_default"](
            x=constant_op.constant(1.),
            y=constant_op.constant(2.))))

  def _make_model_with_tables(self):
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table1_initializer = lookup_ops.KeyValueTensorInitializer(keys, values)
    table1 = lookup_ops.HashTable(table1_initializer, default_val)

    table2_file = self._make_asset("test\nfoo\nbrain\n")
    table2_initializer = lookup_ops.TextFileIdTableInitializer(table2_file)
    table2 = lookup_ops.HashTable(table2_initializer, default_val)

    def _make_lookup_function(table):
      signature = [tensor_spec.TensorSpec(None, dtypes.string)]
      return def_function.function(input_signature=signature)(
          lambda x: table.lookup(x))  # pylint: disable=unnecessary-lambda

    root = tracking.AutoTrackable()
    root.table1 = table1
    root.lookup1 = _make_lookup_function(table1)
    root.table2 = table2
    root.lookup2 = _make_lookup_function(table2)
    return root

  def test_table(self, cycles):
    root = self._make_model_with_tables()
    imported = cycle(root, cycles, signatures={})
    keys = constant_op.constant(["brain", "test", "foo", "surgery"])
    self.assertAllEqual([0, -1, -1, 2], imported.lookup1(keys).numpy())
    self.assertAllEqual([2, 0, 1, -1], imported.lookup2(keys).numpy())

  def test_table_collections_untouched_eager(self, cycles):

    def _gather_nonempty_collections():
      graph = ops.get_default_graph()
      gathered = {}
      for collection in graph.collections:
        collection_contents = graph.get_collection(collection)
        if collection_contents:
          gathered[collection] = collection_contents
      return gathered

    root = self._make_model_with_tables()
    # Warm up collections to ignore those that don't expand every iteration,
    # e.g. the __varscope collection.
    cycle(root, 1)
    original_collections = _gather_nonempty_collections()
    cycle(root, cycles)
    self.assertEqual(original_collections, _gather_nonempty_collections())

  def test_table_in_graph(self, cycles):
    root = self._make_model_with_tables()

    if cycles > 1:
      root = cycle(root, cycles - 1)
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)
    imported = cycle(root, 1)

    with ops.Graph().as_default():
      imported = load.load(path)
      keys = constant_op.constant(["brain", "test", "foo", "surgery"])
      output1 = imported.lookup1(keys)
      output2 = imported.lookup2(keys)
      with monitored_session.MonitoredSession() as sess:
        self.assertAllEqual([0, -1, -1, 2], sess.run(output1))
        self.assertAllEqual([2, 0, 1, -1], sess.run(output2))

  def test_perserve_argspec(self, cycles):
    def f(a, b, c):  # pylint: disable=unused-argument
      return None

    original_fullargspec = tf_inspect.getfullargspec(f)

    root = tracking.AutoTrackable()
    root.f = def_function.function(f)
    imported = cycle(root, cycles)

    restored_fullargspec = tf_inspect.getfullargspec(imported.f)
    self.assertEqual(original_fullargspec, restored_fullargspec)

  def test_canonicalize_inputs(self, cycles):
    @def_function.function(autograph=False)
    def func(a=1, b=2, c=3, training=True):
      if training:
        return [a, b, c, training]
      else:
        return [c, b, a, training]

    # TODO(b/123501567): Work-around to trigger generic traces of a function
    # with extra non tensor args.
    signature = 3*[tensor_spec.TensorSpec(None, dtypes.float32)]
    @def_function.function(input_signature=signature)
    def trigger(a, b, c):
      func(a, b, c, True)
      func(a, b, c, False)

    trigger.get_concrete_function()

    root = tracking.AutoTrackable()
    root.f = func
    root = cycle(root, cycles)
    self.assertAllEqual(root.f(), [1.0, 2.0, 3.0, True])
    self.assertAllEqual(root.f(-1.0, training=False), [3.0, 2.0, -1.0, False])

    with self.assertRaisesRegexp(ValueError,
                                 "Could not find matching function"):
      root.f(["hello", 1.0])

  def test_prefer_specific_trace(self, cycles):
    @def_function.function(autograph=False)
    def func(a):
      if isinstance(a, int):
        return a
      else:
        return a + 1

    self.assertAllEqual(2, func(2).numpy())
    self.assertAllEqual(3, func(constant_op.constant(2)).numpy())

    root = tracking.AutoTrackable()
    root.f = func
    root = cycle(root, cycles)
    self.assertAllEqual(2, root.f(2).numpy())
    self.assertAllEqual(4, root.f(3).numpy())
    self.assertAllEqual(3, root.f(constant_op.constant(2)).numpy())
    self.assertAllEqual(4, root.f(constant_op.constant(3)).numpy())

  def test_partial(self, cycles):
    def f(x, y):
      return x + y

    func = def_function.function(
        functools.partial(f, x=array_ops.zeros([1]), y=array_ops.ones([1])))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(), [1.0])

    root = cycle(root, cycles)
    self.assertAllEqual(root.f(), [1.0])

  def test_partial_with_non_tensor_defaults(self, cycles):

    def f(x, y=3):
      return x + y

    func = def_function.function(functools.partial(f, y=5))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 6)

    root = cycle(root, cycles)
    self.assertAllEqual(root.f(1), 6)

  def test_partial_with_positional(self, cycles):
    def f(x, y):
      return x + y

    func = def_function.function(functools.partial(f, constant_op.constant(5)))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 6)

    root = cycle(root, cycles)
    self.assertAllEqual(root.f(1), 6)

  def test_partial_with_positional_captured_tensors(self, cycles):

    def f(x, y):
      return x + y

    tensor = constant_op.constant(5) + constant_op.constant(7)
    func = def_function.function(functools.partial(f, tensor))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertAllEqual(root.f(1), 13)

    root = cycle(root, cycles)
    self.assertAllEqual(root.f(1), 13)

  def test_partial_keyword_hiding_default(self, cycles):

    def f(x=3, training=True, y=7):
      if training:
        return x + y
      else:
        return x + y + 2

    func = def_function.function(functools.partial(f, y=6))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertEqual(root.f().numpy(), 9)
    self.assertEqual(root.f(training=False).numpy(), 11)

    root = cycle(root, cycles)
    self.assertEqual(root.f().numpy(), 9)
    self.assertEqual(root.f(training=False).numpy(), 11)

  def test_partial_with_kwargs(self, cycles):

    def f(a, b, *args, **kwargs):
      args_sum = sum(args)
      return a + b + kwargs["some_tensor"] * kwargs["learning_rate"] + args_sum

    constant_tensor = constant_op.constant(10)
    func = def_function.function(
        functools.partial(
            f, 7, 1, 2, learning_rate=3, some_tensor=constant_tensor))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertEqual(root.f(constant_op.constant(4)).numpy(), 44)

    root = cycle(root, cycles)
    self.assertEqual(root.f(constant_op.constant(5)).numpy(), 45)

  def test_partial_bind_only_first_argument(self, cycles):
    if sys.version_info[0] < 3:
      self.skipTest("Test is only valid in python3. Only then we get some more "
                    "advanced inspection of partials where this is allowed.")

    def f(x, y):
      return x + y

    partial_func = functools.partial(f, x=5)
    tf_func = def_function.function(partial_func)

    root = tracking.AutoTrackable()
    root.f = tf_func
    self.assertAllEqual(root.f(y=constant_op.constant(7)), 12)

    root = cycle(root, cycles)
    self.assertAllEqual(root.f(y=constant_op.constant(9)), 14)

  def test_partial_with_passed_fn_as_default(self, cycles):

    def f(x, y):
      return x(3) + y

    def my_func(a):
      return 2 * a

    func = def_function.function(functools.partial(f, my_func))

    root = tracking.AutoTrackable()
    root.f = func
    self.assertEqual(root.f(constant_op.constant(3)).numpy(), 9)

    root = cycle(root, cycles)
    self.assertEqual(root.f(constant_op.constant(3)).numpy(), 9)

  def test_partial_with_input_signature(self, cycles):

    def full_function(a, b, c=3.0):
      return a, b, c

    partial = functools.partial(full_function, 1, c=4)
    self.assertAllEqual((1, 2.0, 4), partial(2.0))

    signature = [tensor_spec.TensorSpec([], dtypes.float32)]
    func = def_function.function(partial, input_signature=signature)

    root = tracking.AutoTrackable()
    root.f = func
    a, b, c = root.f(2.0)
    self.assertAllEqual([a.numpy(), b.numpy(), c.numpy()], (1, 2.0, 4))

    root = cycle(root, cycles)
    a, b, c = root.f(3.0)
    self.assertAllEqual([a.numpy(), b.numpy(), c.numpy()], (1, 3.0, 4))

  def test_convert_to_input_signature(self, cycles):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.int32)])
    def func(x):
      return x

    root = tracking.AutoTrackable()
    root.f = func

    root = cycle(root, cycles)

    self.assertEqual([2], root.f([2]).numpy())

  def test_named_tuple(self, cycles):

    class NamedTupleType(collections.namedtuple("NamedTupleType", ["a", "b"])):
      pass

    @def_function.function
    def f(x):
      return x.a + x.b

    f.get_concrete_function(
        NamedTupleType(
            a=tensor_spec.TensorSpec(None, dtypes.float32, name="a"),
            b=tensor_spec.TensorSpec(None, dtypes.float32, name="b")))
    obj = tracking.AutoTrackable()
    obj.__call__ = f
    if sys.version_info.major == 3 and sys.version_info.minor < 5:
      # TODO(allenl): figure out why this doesn't work in Python3.4
      self.skipTest("Not working in Python 3.4")
    imported = cycle(obj, cycles)
    self.assertAllClose(3.,
                        imported(NamedTupleType(a=constant_op.constant(1.),
                                                b=constant_op.constant(2.))))

  def test_extra_args(self, cycles):

    @def_function.function
    def f(x):
      return math_ops.add(x["a"], 1.)
    # Trigger a trace.
    f({"a": constant_op.constant(2.0)})

    obj = tracking.AutoTrackable()
    obj.__call__ = f
    imported = cycle(obj, cycles)

    self.assertEqual(4.0, imported({"a": 3.0}).numpy())

    with self.assertRaisesRegexp(ValueError,
                                 "Could not find matching function to call"):
      imported({"a": 2.0, "b": 3.0})

  def test_shapes_available(self, cycles):

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec([None, 3], dtypes.int32),
        tensor_spec.TensorSpec([None, 2], dtypes.int32)
    ])
    def func(x, y):
      return array_ops.concat([x, y], axis=1)

    root = tracking.AutoTrackable()
    root.f = func

    root = cycle(root, cycles)

    imported_graph = root.f.get_concrete_function().graph
    input_x, input_y = imported_graph.inputs
    self.assertEqual([None, 3], input_x.shape.as_list())
    self.assertEqual([None, 2], input_y.shape.as_list())
    output, = imported_graph.outputs
    self.assertEqual([None, 5], output.shape.as_list())
    signature = root.signatures["serving_default"]
    self.assertEqual(
        [None, 3], signature.inputs[0].shape.as_list())
    self.assertEqual(
        [None, 2], signature.inputs[1].shape.as_list())
    self.assertEqual(
        [None, 5], signature.outputs[0].shape.as_list())

  def test_variables_destroyed(self, cycles):
    v1 = variables.Variable(1.)
    weak_v1 = weakref.ref(v1)
    root = util.Checkpoint(v=v1)
    root = cycle(root, cycles)
    del v1
    self.assertIsNone(weak_v1())
    weak_v2 = weakref.ref(root.v)
    del root
    self.assertIsNone(weak_v2())

  def test_variable_attributes_preserved(self, cycles):
    v = variables.Variable(
        1.,
        trainable=False,
        synchronization=variables.VariableSynchronization.NONE,
        aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)
    self.assertEqual(variables.VariableSynchronization.NONE,
                     v.synchronization)
    self.assertEqual(variables.VariableAggregation.ONLY_FIRST_REPLICA,
                     v.aggregation)
    root = tracking.AutoTrackable()
    root.v = v
    root = cycle(root, cycles)
    self.assertEqual(False, root.v.trainable)
    self.assertEqual(variables.VariableSynchronization.NONE,
                     root.v.synchronization)
    self.assertEqual(variables.VariableAggregation.ONLY_FIRST_REPLICA,
                     root.v.aggregation)

  def test_captured_dataset(self, cycles):

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
    self.assertEqual(
        3 * (1 + 4 + 9 + 16),
        root(constant_op.constant(3, dtype=dtypes.int64)).numpy())
    root = cycle(root, cycles)
    self.assertEqual(
        3 * (1 + 4 + 9 + 16),
        root(constant_op.constant(3, dtype=dtypes.int64)).numpy())

  def test_tuple_signature(self, cycles):
    root = util.Checkpoint()
    root.f = def_function.function(
        lambda: (array_ops.ones([]), array_ops.zeros([])),
        input_signature=())
    for _ in range(cycles):
      root = cycle(root, 1, signatures=root.f)
    self.assertEqual(({"output_0": 1., "output_1": 0.}),
                     self.evaluate(root.signatures["serving_default"]()))

  def test_model_with_custom_function_attached(self, cycles):
    root = util.Checkpoint(model=sequential.Sequential([core.Dense(2)]))

    @def_function.function
    def _use_sequential(x):
      return root.model.call(x)

    root.model.traced_call = _use_sequential

    original = root.model.traced_call(array_ops.zeros([1, 1])).numpy()
    root = cycle(root, cycles)
    self.assertAllEqual(
        original,
        root.model.traced_call(array_ops.zeros([1, 1])).numpy())

  def test_version_info(self, cycles):
    root = util.Checkpoint()
    root = cycle(root, cycles)
    self.assertEqual(versions.__version__, root.tensorflow_version)
    self.assertEqual(versions.__git_version__, root.tensorflow_git_version)

  def test_load_grad_save(self, cycles):
    root = util.Checkpoint()
    root.v = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v * x)
    root.g = def_function.function(root.f)
    for _ in range(cycles):
      with backprop.GradientTape() as tape:
        inp = constant_op.constant(2.)
        tape.watch(inp)
        output = root.g(inp)
        self.assertAllClose(4., output)
      self.assertAllClose(2., tape.gradient(output, inp))
      root = cycle(root, 1)

  def test_destroy_resource(self, cycles):

    def get_handle():
      return gen_resource_variable_ops.var_handle_op(
          shape=tensor_shape.as_shape([]),
          dtype=dtypes.float32,
          shared_name="my_var_name",
          name="my_var",
          container="my_container")

    class MyResourceDeleter(tracking.CapturableResourceDeleter):

      def destroy_resource(self):
        handle = get_handle()
        gen_resource_variable_ops.destroy_resource_op(
            handle, ignore_lookup_error=True)

    class MyResource(tracking.TrackableResource):

      def __init__(self):
        # Set the resource deleter, so when the resource object goes out of
        # scope it will be deleted automatically.
        super(MyResource, self).__init__(deleter=MyResourceDeleter())

      def _create_resource(self):
        return get_handle()

      def _initialize(self):
        gen_resource_variable_ops.assign_variable_op(
            self.resource_handle, 1.0, name="assign")

    class MyModel(tracking.AutoTrackable):

      def __init__(self):
        super(MyModel, self).__init__()
        self.resource = MyResource()

      @def_function.function(input_signature=[])
      def increase(self):
        handle = self.resource.resource_handle
        gen_resource_variable_ops.assign_add_variable_op(
            handle, 10.0, name="assign_add")
        return gen_resource_variable_ops.read_variable_op(
            handle, dtypes.float32)

    root = MyModel()
    imported = cycle(root, cycles)
    self.assertEqual(11, imported.increase().numpy())  # Create the resource.

    handle = imported.resource.resource_handle

    # Delete the imported SaveModel. Since we explicitly set the deleter, it
    # should destroy the resource automatically.
    del imported

    # Try to destroy the resource again, should fail.
    with self.assertRaisesRegexp(errors.NotFoundError,
                                 r"Resource .* does not exist."):
      gen_resource_variable_ops.destroy_resource_op(
          handle, ignore_lookup_error=False)

  def test_function_called_as_operation(self, cycles):

    @framework_function.Defun(dtypes.float32)
    def inner(x):
      return x + 1.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.float32)])
    def outer(x):
      return inner(x)

    root = module.Module()
    root.f = outer
    imported = cycle(root, cycles)
    self.assertAllClose(2., imported.f(constant_op.constant(1.)))

  def test_ragged(self, cycles):

    @def_function.function(input_signature=[
        ragged_tensor.RaggedTensorSpec(shape=[None, None], dtype=dtypes.int32)
    ])
    def f(x):
      return x + 1

    obj = tracking.AutoTrackable()
    obj.f = f

    imported1 = cycle(obj, cycles, signatures={})
    rt = ragged_factory_ops.constant([[1, 2], [3]])
    self.assertAllEqual(imported1.f(rt), [[2, 3], [4]])

    imported2 = cycle(obj, cycles)
    rt = ragged_factory_ops.constant([[1, 2], [3]])
    self.assertAllEqual(imported2.f(rt), [[2, 3], [4]])

@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
@parameterized.named_parameters(
    dict(testcase_name="ReloadOnce", cycles=1),
    dict(testcase_name="ReloadTwice", cycles=2),
    dict(testcase_name="ReloadThrice", cycles=3))
class KerasLoadTest(test.TestCase, parameterized.TestCase):

  def test_dense_features_layer(self, cycles):
    columns = [
        feature_column_lib.numeric_column("x"),
        feature_column_lib.numeric_column("y")
    ]
    layer = feature_column_lib.DenseFeatures(columns)
    model = sequential.Sequential([layer])
    model_input = {"x": constant_op.constant([[1.]]),
                   "y": constant_op.constant([[2.]])}
    self.assertAllClose([[1., 2.]], model.predict(model_input, steps=1))
    loaded = cycle(model, cycles)
    output, = loaded._default_save_signature(model_input).values()
    self.assertAllClose([[1., 2.]], output)
    signature_output, = loaded.signatures["serving_default"](
        **model_input).values()
    self.assertAllClose([[1., 2.]], signature_output)

  def test_dense_features_layer_fit(self, cycles):
    columns = [feature_column_lib.numeric_column("x")]
    model = sequential.Sequential(
        [feature_column_lib.DenseFeatures(columns),
         core.Dense(1)])
    model_input = {"x": constant_op.constant([[1.]])}
    model.compile(optimizer="adam", loss="mse", run_eagerly=True,
                  experimental_run_tf_function=True)
    model.fit(model_input, constant_op.constant([[3.]]))
    loaded = cycle(model, cycles)
    loaded._default_save_signature(model_input)
    loaded.signatures["serving_default"](**model_input)

  def test_multi_output_layer(self, cycles):

    inp = input_layer.Input(name="inp", shape=(None,), dtype=dtypes.float32)

    class _MultiOutput(base_layer.Layer):

      def call(self, x):
        return x + 1., x + 2.

    out = _MultiOutput(name="out")(inp)
    model = training_lib.Model(inp, out)
    loaded = cycle(model, cycles)
    self.assertAllClose(
        dict(out=2., out_1=3.),
        loaded.signatures["serving_default"](constant_op.constant(1.)))

  def test_functional_model_with_conv(self, cycles):
    x = input_layer.Input(name="x", shape=(None, None, 3), dtype=dtypes.float32)
    conved = convolutional.Conv2D(filters=3, kernel_size=3, dilation_rate=2)(x)
    model = training_lib.Model([x], conved)
    model_input = array_ops.ones((1, 10, 10, 3))
    initial_output = model.predict([model_input])
    model = cycle(model, cycles)
    self.assertAllClose(
        [initial_output],
        list(model.signatures["serving_default"](model_input).values()))


class SingleCycleTests(test.TestCase, parameterized.TestCase):

  def test_load_with_tags(self):
    root = tracking.AutoTrackable()
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)
    with self.assertRaises(ValueError):
      load.load(path, tags=[tag_constants.EVAL])
    load.load(path, tags=[tag_constants.SERVING])
    load.load(path, tags=tag_constants.SERVING)
    load.load(path, tags=set([tag_constants.SERVING]))

  def test_docstring_examples(self):
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    exported = util.Checkpoint(v=variables.Variable(3.))
    exported.f = def_function.function(
        lambda x: exported.v * x,
        input_signature=[
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32)])
    save.save(exported, path)
    imported = load.load(path)
    self.assertEqual(3., imported.v.numpy())
    self.assertEqual(6., imported.f(x=constant_op.constant(2.)).numpy())

    save.save(exported, path, exported.f.get_concrete_function())
    imported = load.load(path)
    f = imported.signatures["serving_default"]
    self.assertAllEqual(
        [[-3.]],
        f(x=constant_op.constant([[-1.]]))["output_0"].numpy())


  def test_object_with_extra_dependencies(self):

    class Extra(tracking.AutoTrackable):

      def _list_extra_dependencies_for_serialization(self, cache):
        if self not in cache:
          cache[self] = {"a": variables.Variable(5.)}
        return cache[self]
    root = Extra()
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(root, path)
    imported = load.load(path)
    self.assertEqual(5, self.evaluate(imported.a))

    root.a = variables.Variable(3.)
    with self.assertRaisesRegexp(
        ValueError,
        "object has an attribute named a, which is reserved."):
      save.save(root, path)


if __name__ == "__main__":
  test.main()
