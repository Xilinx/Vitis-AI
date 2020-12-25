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
"""Tests for while_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compat import compat
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.while_v2 import while_loop as while_loop_v2
from tensorflow.python.platform import test


def random_gamma(shape):  # pylint: disable=invalid-name
  return random_ops.random_gamma(shape, 1.0)


def random_gamma_with_alpha_beta(shape):  # pylint: disable=invalid-name
  return random_ops.random_gamma(
      shape, alpha=[[1.], [3.], [5.], [6.]], beta=[[3., 4.]])


def random_poisson_v2(shape):  # pylint: disable=invalid-name
  return random_ops.random_poisson_v2(shape, 1.0)


def random_poisson_v2_with_lam(shape):  # pylint: disable=invalid-name
  return random_ops.random_poisson_v2(shape, [12.2, 3.3])


def fill(shape):  # pylint: disable=invalid-name
  return array_ops.fill(shape, 1.0)


class WhileV2Test(test.TestCase, parameterized.TestCase):

  @test_util.run_deprecated_v1
  def testSingleLoopVar(self):
    x = constant_op.constant(2.)
    ret = while_loop_v2(
        lambda v: v < 8., lambda v: v * v, [x], return_same_structure=False)
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session():
      self.assertEqual(self.evaluate(ret), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  @test_util.run_v1_only("b/120545219")
  def testReturnSameStructureTrue(self):
    x = constant_op.constant(2.)
    ret = while_loop_v2(
        lambda v: v < 8., lambda v: v * v, [x], return_same_structure=True)
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session() as sess:
      eval_result = sess.run(ret)
      self.assertIsInstance(eval_result, list)
      self.assertLen(eval_result, 1)
      self.assertEqual(16., eval_result[0])
      self.assertSequenceEqual(sess.run(grad), [32.])

  def testVerifyInputOutputTypesMatch(self):

    @def_function.function
    def BuildWhile():
      x = constant_op.constant(1., dtypes.float32)

      def Body(x):
        return math_ops.cast(x, dtypes.float16) + 1

      while_loop_v2(lambda x: x < 10, Body, [x])

    with self.assertRaisesRegexp(
        TypeError,
        r"Loop var Const:0 enters the loop with type <dtype: 'float32'> "
        r"but has type <dtype: 'float16'> after 1 iteration."):
      BuildWhile()

  def testGradientTapeResourceVariable(self):
    with context.eager_mode():
      v = variables.Variable(1.)

      @def_function.function
      def fnWithLoop():  # pylint: disable=invalid-name
        with backprop.GradientTape() as tape:
          _, x = while_loop_v2(
              lambda i, _: i < 2,
              lambda i, x: (i + 1, x * v),
              [0, 2.])
        return tape.gradient(x, v)

      self.assertAllEqual(fnWithLoop(), 4.0)

  def testExternalControlDependencies(self):
    with ops.Graph().as_default(), self.test_session():
      v = variables.Variable(1.)
      v.initializer.run()
      op = v.assign_add(1.)

      def body_fn(i):  # pylint: disable=invalid-name
        with ops.control_dependencies([op]):
          return i + 1

      loop = while_loop_v2(lambda i: i < 1, body_fn, [0])
      loop[0].op.run()
      self.assertAllEqual(self.evaluate(v), 2.0)

  @test_util.run_deprecated_v1
  def testMultipleLoopVarsBasic(self):
    x = constant_op.constant(5.)
    y = constant_op.constant(3.)

    # x = 5.
    # y = 3.
    # while x < 45.:
    #   x = x * y
    ret = while_loop_v2(
        lambda v, _: v < 45.,
        lambda v, w: (v * w, w), [x, y],
        return_same_structure=False)
    # ret = [x*y^2, y]

    # Note: This is simply d_ret[0]/d_x since d_ret[1]/d_x is 0.
    grad = gradients_impl.gradients(ret, [x])  # [2*x*y]
    with self.cached_session():
      self.assertSequenceEqual(self.evaluate(ret), [45., 3.])
      self.assertSequenceEqual(self.evaluate(grad), [9.])

  @test_util.run_deprecated_v1
  def testMultipleLoopNonscalarCond(self):
    x = constant_op.constant([[5.]])
    y = constant_op.constant(3.)

    # x = 5.
    # y = 3.
    # while x < 45.:
    #   x = x * y
    ret = while_loop_v2(
        lambda v, _: v < 45.,
        lambda v, w: (v * w, w), [x, y],
        return_same_structure=False)
    # ret == [x*y^2, y]

    # Note: This is simply d_ret[0]/d_x since d_ret[1]/d_x is 0.
    grad = gradients_impl.gradients(ret, [x])  # [2*x*y]
    with self.cached_session():
      self.assertSequenceEqual(self.evaluate(ret), [45., 3.])
      self.assertSequenceEqual(self.evaluate(grad), [9.])

  @test_util.run_deprecated_v1
  def testMultipleLoopVars(self):
    x = constant_op.constant(5.)
    y = constant_op.constant(3.)

    # x = 5.
    # y = 3.
    # while x < 45.:
    #   x = x * y
    #   y = x + y
    ret = while_loop_v2(
        lambda v, _: v < 45.,
        lambda v, w: (v * w, v + w), [x, y],
        return_same_structure=False)
    # ret = [y*x**2 + x*y**2, x*y + x + y]

    gradx_0 = gradients_impl.gradients(ret[0], [x])  # [2*x*y + y**2]
    gradx_1 = gradients_impl.gradients(ret[1], [x])  # [y + 1]
    gradx_2 = gradients_impl.gradients(ret, [x])  # [2*x*y + y**2 + 2*y + 1]
    grady_0 = gradients_impl.gradients(ret[0], [y])  # [2*x*y + x**2]
    grady_1 = gradients_impl.gradients(ret[1], [y])  # [x + 1]
    grady_2 = gradients_impl.gradients(ret, [y])  # [2*x*y + x**2 + x + 1]
    with self.cached_session():
      self.assertSequenceEqual(self.evaluate(ret), [120., 23.])
      self.assertSequenceEqual(self.evaluate(gradx_0), [39.])
      self.assertSequenceEqual(self.evaluate(gradx_1), [4.])
      self.assertSequenceEqual(self.evaluate(gradx_2), [43.])
      self.assertSequenceEqual(self.evaluate(grady_0), [55.])
      self.assertSequenceEqual(self.evaluate(grady_1), [6.])
      self.assertSequenceEqual(self.evaluate(grady_2), [61.])

  @test_util.run_deprecated_v1
  def testGradientTape(self):
    with backprop.GradientTape() as t:
      x = constant_op.constant(2.)
      t.watch(x)
      ret = while_loop_v2(
          lambda v: v < 4., lambda v: v * v, [x],
          return_same_structure=False)  # x**2
    grad = t.gradient(ret, x)
    with self.cached_session() as sess:
      self.assertAllEqual(sess.run(grad), 4.0)

  @test_util.run_deprecated_v1
  def testMultipleWhileLoops(self):
    x = constant_op.constant(2.)
    ret1 = while_loop_v2(
        lambda v: v < 4., lambda v: v * v, [x],
        return_same_structure=False)  # x**2
    ret2 = while_loop_v2(
        lambda v: v < 16., lambda v: v * v, [ret1],
        return_same_structure=False)  # x**4
    grad = gradients_impl.gradients(ret2, [x])  # 4x**3
    grad_grad = gradients_impl.gradients(grad, [x])  # 12x**2
    with self.cached_session():
      self.assertSequenceEqual(self.evaluate(grad), [32.])
      self.assertSequenceEqual(self.evaluate(grad_grad), [48.])

  def testMultipleWhileLoopsWithFunc(self):
    if compat.forward_compatible(2019, 8, 23):
      x = constant_op.constant(2.)

      @def_function.function
      def Fn():
        ret1 = while_loop_v2(
            lambda v: v < 4.,
            lambda v: v * v, [x],
            return_same_structure=False,
            name="while_1")  # x**2
        ret2 = while_loop_v2(
            lambda v: v < 16.,
            lambda v: v * v, [x],
            return_same_structure=False,
            name="while_2")  # x**4
        return ret1, ret2

      concrete_fn = Fn.get_concrete_function()
      while_1 = concrete_fn.graph.get_operation_by_name("while_1")
      while_2 = concrete_fn.graph.get_operation_by_name("while_2")
      self.assertEqual(while_1.type, "StatelessWhile")
      self.assertEqual(while_2.type, "StatelessWhile")
      self.assertEmpty(while_1.control_inputs)
      self.assertEmpty(while_2.control_inputs)

  def testMultipleWhileLoopsWithDeps(self):
    if compat.forward_compatible(2019, 8, 23):
      x = variables.Variable(2.)
      c = constant_op.constant(2.)

      @def_function.function
      def Fn():
        ret1 = while_loop_v2(
            lambda v: v < 4.,
            lambda v: v * x, [c],
            return_same_structure=False,
            name="while_1")  # 2x
        ret2 = while_loop_v2(
            lambda v: v < 16.,
            lambda v: v * x * x, [c],
            return_same_structure=False,
            name="while_2")  # 4x
        return ret1, ret2

      concrete_fn = Fn.get_concrete_function()
      while_1 = concrete_fn.graph.get_operation_by_name("while_1")
      while_2 = concrete_fn.graph.get_operation_by_name("while_2")
      self.assertEqual(while_1.type, "While")
      self.assertEqual(while_2.type, "While")
      self.assertEmpty(while_1.control_inputs)
      self.assertLen(while_2.control_inputs, 1)
      self.assertIs(while_2.control_inputs[0], while_1)

  def testMultipleWhileLoopsWithVarsDeps(self):
    if compat.forward_compatible(2019, 8, 23):
      x1 = variables.Variable(2.)
      x2 = variables.Variable(3.)
      c = constant_op.constant(2.)

      @def_function.function
      def Fn():
        ret1 = while_loop_v2(
            lambda v: v < 4.,
            lambda v: v * x1, [c],
            return_same_structure=False,
            name="while_1")  # 2x
        ret2 = while_loop_v2(
            lambda v: v < 16.,
            lambda v: v * x1 * x1, [c],
            return_same_structure=False,
            name="while_2")  # 4x
        ret3 = while_loop_v2(
            lambda v: v < 4.,
            lambda v: v * x2, [c],
            return_same_structure=False,
            name="while_3")  # 3x
        ret4 = while_loop_v2(
            lambda v: v < 16.,
            lambda v: v * x2 * x2, [c],
            return_same_structure=False,
            name="while_4")  # 9x
        ret5 = while_loop_v2(
            lambda v: v < 16.,
            lambda v: v * v, [c],
            return_same_structure=False,
            name="while_stateless")  # x**2
        return ret1, ret2, ret3, ret4, ret5

      concrete_fn = Fn.get_concrete_function()
      while_1 = concrete_fn.graph.get_operation_by_name("while_1")
      while_2 = concrete_fn.graph.get_operation_by_name("while_2")
      while_3 = concrete_fn.graph.get_operation_by_name("while_3")
      while_4 = concrete_fn.graph.get_operation_by_name("while_4")
      while_stateless = concrete_fn.graph.get_operation_by_name(
          "while_stateless")
      self.assertEqual(while_1.type, "While")
      self.assertEqual(while_2.type, "While")
      self.assertEqual(while_3.type, "While")
      self.assertEqual(while_4.type, "While")
      self.assertEqual(while_stateless.type, "StatelessWhile")
      self.assertEmpty(while_1.control_inputs)
      self.assertLen(while_2.control_inputs, 1)
      self.assertIs(while_2.control_inputs[0], while_1)
      self.assertEmpty(while_3.control_inputs)
      self.assertLen(while_4.control_inputs, 1)
      self.assertIs(while_4.control_inputs[0], while_3)
      self.assertEmpty(while_stateless.control_inputs)

  @test_util.run_deprecated_v1
  def testDoubleDerivative(self):
    x = constant_op.constant(2.)
    ret = while_loop_v2(
        lambda v: v < 8., lambda v: v**2, [x],
        return_same_structure=False)  # x**4
    grad = gradients_impl.gradients(ret, [x])  # 4x**3
    grad_grad = gradients_impl.gradients(grad, [x])  # 12x**2
    with self.cached_session():
      self.assertEqual(self.evaluate(ret), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])
      self.assertSequenceEqual(self.evaluate(grad_grad), [48.])

  @test_util.run_v2_only
  def testMultipleWhileLoopsEager(self):

    @def_function.function
    def Func():
      x = constant_op.constant(2.)
      ret1 = while_loop_v2(
          lambda v: v < 4., lambda v: v * v, [x],
          return_same_structure=False)  # x**2
      ret2 = while_loop_v2(
          lambda v: v < 16.,
          lambda v: v * v, [ret1],
          return_same_structure=False)  # x**4
      grad = gradients_impl.gradients(ret2, [x])[0]  # 4x**3
      grad_grad = gradients_impl.gradients(grad, [x])[0]  # 12x**2
      return grad, grad_grad

    grad, grad_grad = Func()
    self.assertEqual(grad.numpy(), 32.)
    self.assertEqual(grad_grad.numpy(), 48.)

  @test_util.run_v2_only
  def testDoubleDerivativeEager(self):

    @def_function.function
    def Func():
      x = constant_op.constant(2.)
      ret = while_loop_v2(
          lambda v: v < 8., lambda v: v**2, [x],
          return_same_structure=False)  # x**4
      grad = gradients_impl.gradients(ret, [x])[0]  # 4x**3
      grad_grad = gradients_impl.gradients(grad, [x])[0]  # 12x**2
      return ret, grad, grad_grad

    ret, grad, grad_grad = Func()
    self.assertEqual(ret.numpy(), 16.)
    self.assertEqual(grad.numpy(), 32.)
    self.assertEqual(grad_grad.numpy(), 48.)

  def _testPruning(self):
    x = constant_op.constant(1)

    tensor_list = list_ops.empty_tensor_list(
        element_dtype=x.dtype, element_shape=x.shape)

    def Cond(x, tl):
      del tl  # Unused for Cond.
      return x < 5

    def Body(x, tl):
      return x + 1, list_ops.tensor_list_push_back(tl, x)

    outputs = control_flow_ops.while_loop(Cond, Body, [x, tensor_list])

    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(outputs[0])

    g = GetOptimizedGraph()
    # TODO(b/136034023): while_v2 adds an extra loop_counter which is not pruned
    # away, causing an extra Enter node.
    enter_count = 2 if control_flow_util.ENABLE_CONTROL_FLOW_V2 else 1
    self.assertLen([n for n in g.node if n.op == "Enter"], enter_count)
    # Test that the TensorList is pruned out.
    self.assertEmpty([
        n for n in g.node if n.op == "Enter" and
        n.attr["T"].type == dtypes.variant.as_datatype_enum
    ])
    self.assertEmpty([n for n in g.node if n.op == "TensorListPushBack"])

    stack = list_ops.tensor_list_stack(outputs[1], element_dtype=x.dtype)
    train_op.append(stack)
    g = GetOptimizedGraph()
    # TODO(b/136034023): while_v2 adds an extra loop_counter which is not pruned
    # away, causing an extra Enter node.
    enter_count = 3 if control_flow_util.ENABLE_CONTROL_FLOW_V2 else 2
    self.assertLen([n for n in g.node if n.op == "Enter"], enter_count)
    # Test that the TensorList is not pruned out.
    self.assertNotEmpty([
        n for n in g.node if n.op == "Enter" and
        n.attr["T"].type == dtypes.variant.as_datatype_enum
    ])
    self.assertNotEmpty([n for n in g.node if n.op == "TensorListPushBack"])

  @test_util.run_deprecated_v1
  def testPruningV1(self):
    self._testPruning()

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  def testPruningV2(self):
    self._testPruning()

  def _testDoNotAccumulateInvariants(self):
    push_op = ("TensorListPushBack"
               if control_flow_v2_toggles.control_flow_v2_enabled() else
               "StackPushV2")

    # Tests that loop invariants, i.e., tensors that are "captured" by the
    # while loop and not passed as loop variables are not accumulated in
    # gradient computation.
    v = constant_op.constant(5.0, name="v")

    r = control_flow_ops.while_loop(
        lambda _: True, lambda x: v * x, [1.0], maximum_iterations=5)

    output = gradients_impl.gradients(r, v)[0]
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(output)

    g = GetOptimizedGraph()
    # The gradient for v * x requires the value of both v and x. Since v is a
    # loop invariant it is not accumulated so we have just one accumulator for
    # x.
    self.assertLen([n for n in g.node if n.op == push_op], 1)

  @test_util.run_deprecated_v1
  def testDoNotAccumulateInvariantsV1(self):
    self._testDoNotAccumulateInvariants()

  @test_util.run_deprecated_v1
  @test_util.enable_control_flow_v2
  def testDoNotAccumulateInvariantsV2(self):
    self._testDoNotAccumulateInvariants()

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testPruningNested(self):
    assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    x = constant_op.constant(0)

    tensor_list = list_ops.empty_tensor_list(
        element_dtype=x.dtype, element_shape=x.shape)

    def Cond(x, tl):
      del tl  # Unused for Cond.
      return x < 25

    def Body(x, tl):

      def InnerCond(inner_x, unused_outer_x, unused_tl):
        return inner_x < 5

      def InnerBody(inner_x, outer_x, tl):
        return inner_x + 1, outer_x + 1, list_ops.tensor_list_push_back(tl, x)

      inner_x = constant_op.constant(0)
      return control_flow_ops.while_loop(InnerCond, InnerBody,
                                         [inner_x, x, tl])[1:]

    outputs = control_flow_ops.while_loop(Cond, Body, [x, tensor_list])

    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(outputs[0])

    g = GetOptimizedGraph()
    # TODO(b/136034023): while_v2 adds an extra loop_counter which is not pruned
    # away, causing an extra Enter node.
    # enter_count = 4 if control_flow_util.ENABLE_CONTROL_FLOW_V2 else 2
    # self.assertLen([n for n in g.node if n.op == "Enter"], enter_count)
    # Test that the TensorList is pruned out.
    self.assertEmpty([
        n for n in g.node if n.op == "Enter" and
        n.attr["T"].type == dtypes.variant.as_datatype_enum
    ])
    self.assertEmpty([n for n in g.node if n.op == "TensorListPushBack"])
    self.assertEmpty([n for n in g.node if n.op == "_While"])

    stack = list_ops.tensor_list_stack(outputs[1], element_dtype=x.dtype)
    train_op.append(stack)
    g = GetOptimizedGraph()
    # TODO(b/136034023): while_v2 adds an extra loop_counter which is not pruned
    # away, causing an extra Enter node.
    # enter_count = 3 if control_flow_util.ENABLE_CONTROL_FLOW_V2 else 2
    # self.assertLen([n for n in g.node if n.op == "Enter"], enter_count)
    # Test that the TensorList is not pruned out.
    self.assertNotEmpty([
        n for n in g.node if n.op == "Enter" and
        n.attr["T"].type == dtypes.variant.as_datatype_enum
    ])
    self.assertNotEmpty([n for n in g.node if n.op == "TensorListPushBack"])

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testPruningNested2(self):
    assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    v = constant_op.constant(5.0, name="v")

    p = array_ops.placeholder(dtype=dtypes.int32)

    def MidBodyBuilder(iterations):

      def MidBody(i, x):
        r = control_flow_ops.while_loop(
            lambda *_: True,
            lambda i, x: (i + 1, math_ops.multiply(v, x, name="my_mul")),
            (0, x),
            maximum_iterations=iterations,
            name="inner")
        return (i + 1, gradients_impl.gradients(x + r[1], v)[0])

      return MidBody

    def OuterBody(i, x):
      iterations = array_ops.size(p, name="iterations")
      return (i + 1, x + control_flow_ops.while_loop(
          lambda *_: True,
          MidBodyBuilder(iterations), (0, x),
          maximum_iterations=iterations,
          name="mid")[1])

    def CreateWhileLoop():
      with ops.device("/cpu:0"):
        r = control_flow_ops.while_loop(
            lambda *_: True,
            OuterBody, (0, 1.0),
            maximum_iterations=5,
            name="outer")
        return array_ops.identity(r[1])

    output = CreateWhileLoop()
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(output)

    g = GetOptimizedGraph()
    self.assertLen([n for n in g.node if n.op == "TensorListPushBack"], 1)

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testPruningNested3(self):
    assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    v = constant_op.constant(5.0, name="v")

    def CreateWhileLoop():
      r = control_flow_ops.while_loop(
          lambda _: True,
          lambda x: math_ops.multiply(v, x, name="my_mul"), [1.0],
          maximum_iterations=5,
          name="outer")
      return array_ops.identity(r)

    r = CreateWhileLoop()
    output = gradients_impl.gradients(r, v)[0]
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(output)

    g = GetOptimizedGraph()
    self.assertLen([n for n in g.node if n.op == "TensorListPushBack"], 1)

  def _assertNotAccumulated(self, while_op, index):
    """Asserts that `while_op` input at `index` is not accumulated."""
    body_graph = while_v2._get_graph(while_op, "body")
    placeholder = body_graph.inputs[index]
    self.assertNotIn("TensorListPushBack",
                     [op.type for op in placeholder.consumers()])

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testDoNotOutputLoopCounterAsIntermediate(self):
    assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    v = constant_op.constant(5.0, name="v")
    r = control_flow_ops.while_loop(
        lambda _: True, lambda x: v * x, [1.0], maximum_iterations=5)
    # Skip over Identity.
    while_op = r.op.inputs[0].op
    self._assertNotAccumulated(while_op, 0)

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  @test_util.enable_output_all_intermediates
  def testDoNotOutputLoopInvariantAsIntermediate(self):
    assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE

    def GetInputIndex(op, tensor):
      for index, inp in enumerate(op.inputs):
        if inp is tensor:
          return index

    v = constant_op.constant(5.0, name="v")
    r = control_flow_ops.while_loop(
        lambda _: True, lambda x: v * x, [1.0], maximum_iterations=5)
    # Skip over Identity.
    while_op = r.op.inputs[0].op
    # We can't directly use while_op.inputs.index() because Tensors are not
    # hashshable.
    index = GetInputIndex(while_op, v)
    self._assertNotAccumulated(while_op, index)

  @test_util.run_deprecated_v1
  def testCaptureExternalTensorInCond(self):
    x = constant_op.constant(2.)
    y = constant_op.constant(1.)
    ret = while_loop_v2(
        lambda v: v + y < 9.,
        lambda v: v * 3., [x],
        return_same_structure=False)
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session():
      self.assertEqual(self.evaluate(ret), 18.)
      self.assertSequenceEqual(self.evaluate(grad), [9.])

  @test_util.run_deprecated_v1
  def testCaptureExternalTensorInBody(self):
    x = constant_op.constant(2.)
    y = constant_op.constant(3.)
    ret = while_loop_v2(
        lambda v: v < 8., lambda v: v * y, [x], return_same_structure=False)
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session():
      self.assertEqual(self.evaluate(ret), 18.)
      self.assertSequenceEqual(self.evaluate(grad), [9.])

  @test_util.run_deprecated_v1
  def testLoopWithTensorListPushBack(self):
    x = constant_op.constant(2.)

    tensor_list = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=ScalarShape())

    def Cond(x, tl):
      del tl  # Unused for Cond.
      return x < 5.

    def Body(x, tl):
      tl = list_ops.tensor_list_push_back(tl, x)
      tl = list_ops.tensor_list_push_back(tl, constant_op.constant(100.))
      return x**2., tl

    ret = while_loop_v2(
        Cond, Body, [x, tensor_list], return_same_structure=False)
    grad = gradients_impl.gradients(ret[0], x)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(ret[0]), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  @test_util.run_deprecated_v1
  def testDuplicateAccumulator(self):
    x = constant_op.constant(2.)

    tensor_list = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=ScalarShape())

    def Cond(x, tl):
      del tl  # Unused for Cond.
      return x < 5.

    def Body(x, tl):
      # There is an accumulator in the loop already so we should not add
      # another.
      tl = list_ops.tensor_list_push_back(tl, x)
      return x**2., tl

    ret = while_loop_v2(
        Cond, Body, [x, tensor_list], return_same_structure=False)

    for op in ops.get_default_graph().get_operations():
      if op.type == "While" or op.type == "StatelessWhile":
        while_op = op

    body_graph = while_v2._get_graph(while_op, "body")
    x_input_index = [i for i, inp in enumerate(while_op.inputs) if inp == x][0]
    x_input_t = body_graph.inputs[x_input_index]
    accumulator_count = len(
        [c for c in x_input_t.consumers() if c.type == "TensorListPushBack"])
    self.assertEqual(accumulator_count, 1)

    grad = gradients_impl.gradients(ret[0], x)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(ret[0]), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  @parameterized.named_parameters(
      ("UnknownShape", None),
      ("PartiallyDefinedShape", [None, 2]),
      ("FullyDefinedShape", [1, 2]),
  )
  @test_util.run_deprecated_v1
  def testAccumulatorElementShape(self, shape):

    def MatchShape(actual_tensor_shape):
      # Compare the shapes, treating None dimensions as equal. We do not
      # directly check actual_tensor_shape and tf.TensorShape(shape) for
      # equality because tf.Dimension.__eq__ returns None if either dimension is
      # None.
      if shape is None:
        self.assertIsNone(actual_tensor_shape.dims)
      else:
        self.assertListEqual(actual_tensor_shape.as_list(), shape)

    def GetAccumulatorForInputAtIndex(while_op, idx):
      body_graph = while_v2._get_graph(while_op, "body")
      y_input_t = body_graph.inputs[idx]
      push_back_node = [c for c in y_input_t.consumers()
                        if c.type == "TensorListPushBack"][0]
      output_idx = body_graph.outputs.index(push_back_node.outputs[0])
      return while_op.outputs[output_idx]

    x = array_ops.placeholder(dtype=dtypes.float32, shape=shape)
    y = array_ops.placeholder(dtype=dtypes.float32, shape=shape)

    # Forward pass.
    ret = while_loop_v2(lambda v, u: v < 8.,
                        lambda v, u: (math_ops.pow(v, u), u),
                        [x, y],
                        return_same_structure=True)
    while_op = ret[0].op.inputs[0].op
    # Gradient pass.
    grad = gradients_impl.gradients(ret[0], x)
    # Note: There is an Identity b/w grad[0] and the While op.
    grad_while_op = grad[0].op.inputs[0].op

    # Get the TensorList output of While op containing the accumulated values
    # of y.
    x_input_index = [i for i, inp in enumerate(while_op.inputs) if x == inp][0]
    output = GetAccumulatorForInputAtIndex(while_op, x_input_index)
    _, val = list_ops.tensor_list_pop_back(output,
                                           element_dtype=dtypes.float32)
    MatchShape(val.shape)

    # Take second derivative to generate intermediate grad_while_op outputs
    gradients_impl.gradients(grad, x)

    # Get the TensorList output of gradient While op containing the accumulated
    # values of grad_x (note that grad_x is needed by the second derivative).
    # grad_while_op.inputs:
    grad_output_index = grad_while_op.outputs.index(grad[0].op.inputs[0])
    grad_output = GetAccumulatorForInputAtIndex(grad_while_op,
                                                grad_output_index)
    _, val = list_ops.tensor_list_pop_back(grad_output,
                                           element_dtype=dtypes.float32)
    MatchShape(val.shape)

  def _createWhile(self, name):
    """Helper function testDefaultName."""
    output = while_v2.while_loop(
        lambda i: i < 3,
        lambda i: i + 1, [constant_op.constant(0)],
        return_same_structure=False)
    while_op = output.op.inputs[0].op
    if compat.forward_compatible(2019, 8, 23):
      self.assertEqual(while_op.type, "StatelessWhile")
    return while_op

  def testDefaultName(self):
    with ops.Graph().as_default():
      while_op = self._createWhile(None)
      self.assertEqual(while_op.name, "while")
      self.assertRegexpMatches(
          while_op.get_attr("cond").name, r"while_cond_\d*")
      self.assertRegexpMatches(
          while_op.get_attr("body").name, r"while_body_\d*")

    with ops.Graph().as_default():
      with ops.name_scope("foo"):
        while1_op = self._createWhile("")
        self.assertEqual(while1_op.name, "foo/while")
        self.assertRegexpMatches(
            while1_op.get_attr("cond").name, r"foo_while_cond_\d*")
        self.assertRegexpMatches(
            while1_op.get_attr("body").name, r"foo_while_body_\d*")

        while2_op = self._createWhile(None)
        self.assertEqual(while2_op.name, "foo/while_1")
        self.assertRegexpMatches(
            while2_op.get_attr("cond").name, r"foo_while_1_cond_\d*")
        self.assertRegexpMatches(
            while2_op.get_attr("body").name, r"foo_while_1_body_\d*")

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  def testWhileAndTensorArray(self):
    param = constant_op.constant(2.0)
    y0 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="elems")
    # map_fn uses TensorArray internally.
    r = map_fn.map_fn(lambda x: math_ops.multiply(x, param), y0)
    grad = gradients_impl.gradients(r, param)[0]
    self.assertAllClose([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], self.evaluate(r))
    self.assertAllClose(21.0, self.evaluate(grad))

  @test_util.run_deprecated_v1
  def testNestedWhile(self):
    # Compute sum of geometric progression: n^0 + n^1 + ... + n^m
    # We compute the pow using a while loop.
    n = constant_op.constant(3.)
    m = constant_op.constant(5.)
    sum_of_powers = constant_op.constant(0.)

    def Body(i, previous_sum):
      prod = constant_op.constant(1.)
      return i - 1., previous_sum + while_loop_v2(
          lambda c, _: c > 0,
          lambda c, v: (c - 1., v * n), [i, prod],
          return_same_structure=False)[1]

    result = while_loop_v2(
        lambda i, _: i >= 0,
        Body, [m, sum_of_powers],
        return_same_structure=False)[1]
    grad = gradients_impl.gradients(result, [n])
    with self.cached_session() as sess:
      self.assertEqual(self.evaluate(result), 364.)
      self.assertSequenceEqual(self.evaluate(grad), [547.])

  @test_util.run_deprecated_v1
  def testIdentityNodeInBody(self):

    def Body(v):
      v = array_ops.identity(v)
      v = array_ops.identity(v)
      return v * v

    x = constant_op.constant(2.)
    ret = while_loop_v2(
        lambda v: v < 8., Body, [x], return_same_structure=False)
    grad = gradients_impl.gradients(ret, [x])
    with self.cached_session() as sess:
      self.assertEqual(self.evaluate(ret), 16.)
      self.assertSequenceEqual(self.evaluate(grad), [32.])

  @test_util.run_deprecated_v1
  def testForwardPassRewrite(self):
    if compat.forward_compatible(2019, 8, 23):
      x = constant_op.constant(1.0, name="x")
      output = while_v2.while_loop(lambda x: x < 10.0,
                                   lambda x: x * 2.0,
                                   [x])[0]
      while_op = output.op.inputs[0].op
      self.assertEqual(while_op.type, "StatelessWhile")
      # outputs = [loop_counter, max_iters, x]
      self.assertLen(while_op.outputs, 3)

      gradients_impl.gradients(output, x)
      # while_op should have been rewritten to output intermediates.
      # outputs = [loop_counter, max_iters, x, x_accumulator]
      self.assertLen(while_op.outputs, 4)

      gradients_impl.gradients(output, x)
      # Computing the gradient again shouldn't rewrite while_op again.
      self.assertLen(while_op.outputs, 4)

  @parameterized.named_parameters(
      ("RandomUniform", random_ops.random_uniform, [5, 3]),
      ("RandomNormal", random_ops.random_normal, [5, 3]),
      ("ParameterizedTruncatedNormal",
       random_ops.parameterized_truncated_normal, [5, 3]),
      ("TruncatedNormal", random_ops.truncated_normal, [5, 3]),
      ("RandomGamma", random_gamma, [5, 3]),
      ("RandomPoissonV2", random_poisson_v2, [5, 3]),
      ("RandomGammaWithAlphaBeta", random_gamma_with_alpha_beta, [5, 3, 4, 2]),
      ("RandomPoissonV2WithLam", random_poisson_v2_with_lam, [5, 3, 2]),
  )
  @test_util.run_deprecated_v1
  def testRandomOpsShape(self, random_fn, expected_shape):
    shape = constant_op.constant([3])

    def Body(i, u):
      shape_extended = array_ops.concat([[5], shape], axis=0)
      u = random_fn(shape_extended)
      assert u.shape.as_list() == expected_shape, str(u.shape.as_list())
      return i + 1, u

    _, _ = while_loop_v2(
        cond=lambda i, _: i < 3,
        body=Body,
        loop_vars=[
            0,
            array_ops.zeros(expected_shape, dtype=dtypes.float32),
        ])

  @test_util.run_deprecated_v1
  def testReshapeShape(self):
    shape = constant_op.constant([3, 4])

    def Body(i, u):
      shape_extended = array_ops.concat([[5], shape], axis=0)
      u = array_ops.reshape(u, [-1])
      assert u.shape.as_list() == [60], str(u.shape.as_list())
      u = array_ops.reshape(u, shape_extended)
      assert u.shape.as_list() == [5, 3, 4], str(u.shape.as_list())
      return i + 1, u

    _, _ = while_loop_v2(
        cond=lambda i, _: i < 3,
        body=Body,
        loop_vars=[
            0,
            array_ops.zeros([5, 3, 4], dtype=dtypes.float32),
        ])

  @parameterized.named_parameters(
      ("Zeros", array_ops.zeros),
      ("Ones", array_ops.ones),
      ("Fill", fill),
  )
  @test_util.run_deprecated_v1
  def testFillOpsShape(self, fill_fn):
    shape = constant_op.constant([3, 4])

    def Body(i, u):
      shape_extended = array_ops.concat([[5], shape], axis=0)
      u = fill_fn(shape_extended)
      assert u.shape.as_list() == [5, 3, 4], str(u.shape.as_list())
      return i + 1, u

    _, _ = while_loop_v2(
        cond=lambda i, _: i < 3,
        body=Body,
        loop_vars=[
            0,
            array_ops.zeros([5, 3, 4], dtype=dtypes.float32),
        ])

  @test_util.run_deprecated_v1
  def testExternalColocationGrad(self):
    external_t = constant_op.constant(2.)
    v0 = constant_op.constant(2.)

    def Body(v):
      with ops.colocate_with(external_t):
        return v * v

    ret = while_loop_v2(lambda v: v < 8., Body, [v0])[0]
    grad = gradients_impl.gradients(ret, [v0])[0]
    self.assertAllEqual(ret, 16.)
    self.assertAllEqual(grad, 32.)

  @test_util.run_deprecated_v1
  def testDoNotAccumulateConstNodes(self):

    def Body(v):
      return v * 2.0

    v0 = constant_op.constant(2.)
    ret = while_loop_v2(lambda v: v < 8., Body, [v0])[0]
    # Gradients computation has the side-effect of updating the forward op
    # which is what we want to test.
    unused_grad = gradients_impl.gradients(ret, [v0])[0]
    # ret is separated from the `While` op by an `Identity` so we skip over
    # that.
    forward_while_op = ret.op.inputs[0].op
    body_graph = while_v2._get_graph(forward_while_op, "body")
    push_back_nodes = [
        o for o in body_graph.get_operations() if o.type == "TensorListPushBack"
    ]
    # Gradient of `Mul` requires accumulating both its inputs. But since one
    # of those is a Const (2.0), we should have just one accumulator.
    self.assertLen(push_back_nodes, 1)


def ScalarShape():
  return ops.convert_to_tensor([], dtype=dtypes.int32)


def GetOptimizedGraph():
  mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())
  config = config_pb2.ConfigProto()
  config.graph_options.rewrite_options.CopyFrom(
      rewriter_config_pb2.RewriterConfig(
          constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
          memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL))
  return tf_optimizer.OptimizeGraph(config, mg)


if __name__ == "__main__":
  test.main()
