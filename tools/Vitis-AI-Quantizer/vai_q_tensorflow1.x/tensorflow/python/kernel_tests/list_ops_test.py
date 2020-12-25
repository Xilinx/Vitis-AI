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
"""Tests for ops which manipulate lists of tensors."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np  # pylint: disable=unused-import

from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ListOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def _testPushPop(self, max_num_elements):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[],
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    l = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    l, e = self.evaluate((l, e))
    self.assertAllEqual(l, [])
    self.assertAllEqual(e, 1.0)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  def testPushPop(self, max_num_elements):
    self._testPushPop(max_num_elements)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  @test_util.run_gpu_only
  def testPushPopGPU(self, max_num_elements):
    with context.device("gpu:0"):
      self._testPushPop(max_num_elements)

  @test_util.run_deprecated_v1
  def testPushInFullListFails(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[], max_num_elements=1)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "Tried to push item into a full list"):
      l = list_ops.tensor_list_push_back(l, 2.)
      self.evaluate(l)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  @test_util.run_deprecated_v1
  def testPopFromEmptyTensorListFails(self, max_num_elements):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[],
        max_num_elements=max_num_elements)
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "Trying to pop from an empty list"):
      l = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      self.evaluate(l)

  def testPopUninitializedTensorUseListElementShape(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[2, 3], num_elements=3)
    _, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    l = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    l, e = self.evaluate((l, e))
    self.assertAllEqual(e, np.zeros((2, 3)))
    self.assertAllEqual(l, np.zeros((3, 2, 3)))

  def testPopUninitializedTensorUseSpecifiedElementShape(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[None, 3], num_elements=3)
    _, e = gen_list_ops.tensor_list_pop_back(
        l, element_dtype=dtypes.float32, element_shape=[4, 3])
    self.assertAllEqual(e, np.zeros((4, 3)))

  def testPopUninitializedTensorWithInvalidElementShapeFails(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "Trying to read an uninitialized tensor but "
        "element_shape is not fully defined"):
      _, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      self.evaluate(e)

    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[None, 2], num_elements=3)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r"Incompatible shapes during merge: \[1,3\] vs. \[\?,2\]"):
      _, e = gen_list_ops.tensor_list_pop_back(
          l, element_dtype=dtypes.float32, element_shape=[1, 3])
      self.evaluate(e)

  def testPushGetGrad(self):
    with backprop.GradientTape() as tape:
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32, element_shape=None)
      c0 = constant_op.constant(5.0)
      c1 = constant_op.constant([10.0, 20.0])
      tape.watch(c0)
      tape.watch(c1)
      l = list_ops.tensor_list_push_back(l, c0)
      l = list_ops.tensor_list_push_back(l, c1)
      t1 = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(t1), [10.0, 20.0])
      # t1 == c1 so the gradient should be [0., [1., 1.]]
      # This tests that the gradient of push_back correctly converts DT_INVALID
      # tensors to zeros. The list returned by the gradient of GetItem will
      # have only have tensor at index 1 set and others set to DT_INVALID.
      dt0, dt1 = tape.gradient(t1, [c0, c1])
      self.assertAllEqual(self.evaluate(dt1), [1.0, 1.0])
      self.assertEqual(self.evaluate(dt0), 0.0)

  def _testStack(self, max_num_elements):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[],
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    if not context.executing_eagerly():
      self.assertAllEqual(t.shape.as_list(), [None])
    self.assertAllEqual(self.evaluate(t), [1.0, 2.0])

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  def testStack(self, max_num_elements):
    self._testStack(max_num_elements)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  @test_util.run_gpu_only
  def testStackGPU(self, max_num_elements):
    with context.device("gpu:0"):
      self._testStack(max_num_elements)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 3))
  @test_util.run_deprecated_v1
  def testStackWithUnknownElementShape(self, max_num_elements):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=None,
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))

    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [1.0, 2.0])

    # Should raise an error when the element tensors do not all have the same
    # shape.
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "Incompatible ranks during merge: 0 vs. 1"):
      l = list_ops.tensor_list_push_back(l, constant_op.constant([3.0, 4.0]))
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 3))
  @test_util.run_deprecated_v1
  def testStackWithPartiallyDefinedElementShape(self, max_num_elements):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[None],
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant([1.0]))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([2.0]))

    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[1.0], [2.0]])

    # Should raise an error when the element tensors do not all have the same
    # shape.
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r"Incompatible shapes during merge: \[1\] vs. \[2\]"):
      l = list_ops.tensor_list_push_back(l, constant_op.constant([2.0, 3.0]))
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  @test_util.run_deprecated_v1
  def testStackEmptyList(self, max_num_elements):
    # Should be able to stack empty lists with fully defined element_shape.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[1, 2],
        max_num_elements=max_num_elements)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t).shape, (0, 1, 2))

    # Should not be able to stack empty lists with partially defined
    # element_shape.
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "non-fully-defined"):
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=[None, 2],
          max_num_elements=max_num_elements)
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.evaluate(t)

    # Should not be able to stack empty lists with undefined element_shape.
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "non-fully-defined"):
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=None,
          max_num_elements=max_num_elements)
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def _testStackWithUninitializedTensors(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[], num_elements=3)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(t, [0., 0., 0.])

  def testStackWithUninitializedTensors(self):
    self._testStackWithUninitializedTensors()

  @test_util.run_gpu_only
  def testStackWithUninitializedTensorsGpu(self):
    with context.device("gpu:0"):
      self._testStackWithUninitializedTensors()

  def _testStackWithUninitializedTensorsInferShape(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    l = list_ops.tensor_list_set_item(l, 1, [1., 2.])
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(t, [[0., 0.], [1., 2.], [0., 0.]])

  def testStackWithUninitializedTensorsInferShape(self):
    self._testStackWithUninitializedTensorsInferShape()

  @test_util.run_gpu_only
  def testStackWithUninitializedTensorsInferShapeGpu(self):
    with context.device("gpu:0"):
      self._testStackWithUninitializedTensorsInferShape()

  def testStackReservedListWithNoElementsAndPartialElementShapeFails(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "Tried to stack list which only contains "
                                 "uninitialized tensors and has a "
                                 "non-fully-defined element_shape: <unknown>"):
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testStackUsingSpecifiedElementShape(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    t = gen_list_ops.tensor_list_stack(
        l, element_dtype=dtypes.float32, element_shape=[])
    if context.executing_eagerly():
      self.assertEqual(t.shape.as_list(), [3])
    else:
      self.assertEqual(t.shape.as_list(), [None])
    self.assertAllEqual(self.evaluate(t), np.zeros((3,)))

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  def testGatherGrad(self, max_num_elements):
    with backprop.GradientTape() as tape:
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=[],
          max_num_elements=max_num_elements)
      c0 = constant_op.constant(1.0)
      tape.watch(c0)
      l = list_ops.tensor_list_push_back(l, c0)
      l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
      t = list_ops.tensor_list_gather(l, [1, 0], element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(t), [2.0, 1.0])
      s = (t[0] + t[1]) * (t[0] + t[1])
    dt = tape.gradient(s, c0)
    self.assertAllEqual(self.evaluate(dt), 6.0)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 3))
  @test_util.run_deprecated_v1
  def testGatherWithUnknownElementShape(self, max_num_elements):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=None,
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([3.0, 4.0]))

    t = list_ops.tensor_list_gather(l, [1, 0], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [2.0, 1.0])

    t = list_ops.tensor_list_gather(l, [2], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[3.0, 4.0]])

    # Should raise an error when the requested tensors do not all have the same
    # shape.
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "Incompatible ranks during merge: 0 vs. 1"):
      t = list_ops.tensor_list_gather(l, [0, 2], element_dtype=dtypes.float32)
      self.evaluate(t)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 3))
  @test_util.run_deprecated_v1
  def testGatherWithPartiallyDefinedElementShape(self, max_num_elements):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[None],
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant([1.0]))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([2.0, 3.0]))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([4.0, 5.0]))

    t = list_ops.tensor_list_gather(l, [0], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[1.0]])

    t = list_ops.tensor_list_gather(l, [1, 2], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[2.0, 3.0], [4.0, 5.0]])

    # Should raise an error when the requested tensors do not all have the same
    # shape.
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r"Incompatible shapes during merge: \[1\] vs. \[2\]"):
      t = list_ops.tensor_list_gather(l, [0, 2], element_dtype=dtypes.float32)
      self.evaluate(t)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 3))
  @test_util.run_deprecated_v1
  def testGatherEmptyList(self, max_num_elements):
    # Should be able to gather from empty lists with fully defined
    # element_shape.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[1, 2],
        max_num_elements=max_num_elements)
    t = list_ops.tensor_list_gather(l, [], element_dtype=dtypes.float32)
    self.assertAllEqual((0, 1, 2), self.evaluate(t).shape)

    # Should not be able to gather from empty lists with partially defined
    # element_shape.
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "non-fully-defined"):
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=[None, 2],
          max_num_elements=max_num_elements)
      t = list_ops.tensor_list_gather(l, [], element_dtype=dtypes.float32)
      self.evaluate(t)

    # Should not be able to gather from empty lists with undefined
    # element_shape.
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "non-fully-defined"):
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=None,
          max_num_elements=max_num_elements)
      t = list_ops.tensor_list_gather(l, [], element_dtype=dtypes.float32)
      self.evaluate(t)

  def testGatherGradWithNonContiguousIndices(self):
    with backprop.GradientTape(persistent=True) as tape:
      t = constant_op.constant([1.0, 2.0, 3.0])
      l = list_ops.tensor_list_from_tensor(t, element_shape=[])
      c = constant_op.constant(5.0)
      tape.watch(c)
      l = list_ops.tensor_list_set_item(l, 1, c)
      t = list_ops.tensor_list_gather(l, [1], element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(t), [5.0])
      s = t[0] * t[0]
    dt = tape.gradient(s, c)
    self.assertAllEqual(self.evaluate(dt), 10.0)
    dl = tape.gradient(t, l)
    dl_length = list_ops.tensor_list_length(dl)
    self.assertAllEqual(self.evaluate(dl_length), 3)

  def _testGatherWithUninitializedTensors(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[], num_elements=3)
    t = list_ops.tensor_list_gather(l, [0, 2], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [0., 0.])

  def testGatherWithUninitializedTensors(self):
    self._testGatherWithUninitializedTensors()

  @test_util.run_gpu_only
  def testGatherWithUninitializedTensorsGpu(self):
    with context.device("gpu:0"):
      self._testGatherWithUninitializedTensors()

  def _testGatherWithUninitializedTensorsInferShape(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    l = list_ops.tensor_list_set_item(l, 1, [1., 2.])
    t = list_ops.tensor_list_gather(l, [1, 2], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[1., 2.], [0., 0.]])

  def testGatherWithUninitializedTensorsInferShape(self):
    self._testGatherWithUninitializedTensorsInferShape()

  @test_util.run_gpu_only
  def testGatherWithUninitializedTensorsInferShapeGpu(self):
    with context.device("gpu:0"):
      self._testGatherWithUninitializedTensorsInferShape()

  def testGatherReservedListWithNoElementsAndPartialElementShapeFails(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "Tried to gather uninitialized tensors from a"
        " list with non-fully-defined element_shape"):
      t = list_ops.tensor_list_gather(l, [0], element_dtype=dtypes.float32)
      self.evaluate(t)

  def testGatherUsingSpecifiedElementShape(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    t = gen_list_ops.tensor_list_gather(
        l, [0, 1, 2], element_dtype=dtypes.float32, element_shape=[])
    self.assertEqual(t.shape.as_list(), [3])
    self.assertAllEqual(self.evaluate(t), np.zeros((3,)))

  def testScatterOutputListSize(self):
    c0 = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_scatter(c0, [1, 3], [])
    # TensorListScatter should return a list with size largest index + 1.
    self.assertAllEqual(list_ops.tensor_list_length(l), 4)

  def testScatterOutputListSizeWithNumElementsSpecified(self):
    c0 = constant_op.constant([1.0, 2.0])
    l = gen_list_ops.tensor_list_scatter_v2(
        c0, [1, 3], list_ops._build_element_shape([]), num_elements=5)
    # TensorListScatter should return a list with size num_elements.
    self.assertAllEqual(list_ops.tensor_list_length(l), 5)

  def testScatterFailsWhenIndexLargerThanNumElements(self):
    c0 = constant_op.constant([1.0, 2.0])
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "TensorListScatter: Trying to scatter at index 3 in list with size 3"):
      l = gen_list_ops.tensor_list_scatter_v2(
          c0, [1, 3], list_ops._build_element_shape([]), num_elements=3)
      self.evaluate(l)

  def testScatterFailsWithInvalidNumElements(self):
    c0 = constant_op.constant([1.0, 2.0])
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "TensorListScatter expects num_elements >= -1, found: -2"):
      l = gen_list_ops.tensor_list_scatter_v2(
          c0, [1, 3], list_ops._build_element_shape([]), num_elements=-2)
      self.evaluate(l)

  def testScatterWithInvalidRowsInInputTensorFails(self):
    c0 = constant_op.constant([1.0, 2.0])
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "Invalid number of rows in input tensor. Expected: 3 Actual: 2"):
      l = list_ops.tensor_list_scatter(c0, [1, 0, 2], [])
      self.evaluate(l)

  def testScatterWithNegativeIndicesFails(self):
    c0 = constant_op.constant([1.0, 2.0])
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "Indices in TensorListScatter must all be non-negative."):
      l = list_ops.tensor_list_scatter(c0, [-1, -2], element_shape=[])
      self.evaluate(l)

  def testScatterIntoExistingList(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[], num_elements=3)
    l = list_ops.tensor_list_scatter(tensor=[1.], indices=[0], element_shape=[])
    l = list_ops.tensor_list_scatter(
        tensor=[2., 3.], indices=[1, 2], element_shape=[], input_handle=l)
    self.assertAllEqual(
        list_ops.tensor_list_stack(l, element_dtype=dtypes.float32),
        [1., 2., 3.])

  def testScatterGrad(self):
    with backprop.GradientTape() as tape:
      c0 = constant_op.constant([1.0, 2.0])
      tape.watch(c0)
      l = list_ops.tensor_list_scatter(c0, [1, 0], element_shape=[])
      t0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      t1 = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(t0), 2.0)
      self.assertAllEqual(self.evaluate(t1), 1.0)
      loss = t0 * t0 + t1 * t1
    dt = tape.gradient(loss, c0)
    self.assertAllEqual(self.evaluate(dt), [2., 4.])

  def testScatterWithPartialReadGrad(self):
    with backprop.GradientTape() as tape:
      c0 = constant_op.constant([1.0, 2.0])
      tape.watch(c0)
      l = list_ops.tensor_list_scatter(c0, [1, 0], element_shape=[])
      t0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(t0), 2.0)
      loss = t0 * t0
    dt = tape.gradient(loss, c0)
    self.assertAllEqual(self.evaluate(dt), [0., 4.])

  def testTensorListFromTensor(self):
    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(t, element_shape=[])
    e = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
    self.assertAllEqual(e, 1.0)
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(e, 2.0)
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(e, 1.0)
    self.assertAllEqual(list_ops.tensor_list_length(l), 0)

  @test_util.run_gpu_only
  def testFromTensorGPU(self):
    with context.device("gpu:0"):
      self.testTensorListFromTensor()

  def testGetSet(self):
    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(t, element_shape=[])
    e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e0), 1.0)
    l = list_ops.tensor_list_set_item(l, 0, 3.0)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [3.0, 2.0])

  @test_util.run_gpu_only
  def testGetSetGPU(self):
    with context.device("gpu:0"):
      self.testGetSet()

  def testGetSetReserved(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[], num_elements=2)
    e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
    self.assertAllEqual(e0, 0.0)
    l = list_ops.tensor_list_set_item(l, 0, 3.0)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(t, [3.0, 0.0])

  @test_util.run_gpu_only
  def testGetSetReservedGPU(self):
    with context.device("gpu:0"):
      self.testGetSetReserved()

  def testSetGetGrad(self):
    with backprop.GradientTape() as tape:
      t = constant_op.constant(5.)
      tape.watch(t)
      l = list_ops.tensor_list_reserve(
          element_dtype=dtypes.float32, element_shape=[], num_elements=3)
      l = list_ops.tensor_list_set_item(l, 1, 2. * t)
      e = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(e), 10.0)
    self.assertAllEqual(self.evaluate(tape.gradient(e, t)), 2.0)

  def testGetUninitializedTensorUseListElementShape(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[], num_elements=3)
    l = list_ops.tensor_list_set_item(l, 0, 5.)
    e1 = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
    e2 = list_ops.tensor_list_get_item(l, 2, element_dtype=dtypes.float32)
    self.assertEqual(self.evaluate(e1), 0.)
    self.assertEqual(self.evaluate(e2), 0.)

  def testGetUninitializedTensorUseSpecifiedElementShape(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    e0 = gen_list_ops.tensor_list_get_item(
        l, 0, element_shape=[], element_dtype=dtypes.float32)
    e1 = gen_list_ops.tensor_list_get_item(
        l, 1, element_shape=[2, 3], element_dtype=dtypes.float32)
    self.assertEqual(e0.shape.as_list(), [])
    self.assertEqual(e1.shape.as_list(), [2, 3])
    self.assertEqual(self.evaluate(e0), 0.)
    self.assertAllEqual(self.evaluate(e1), np.zeros((2, 3)))

    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[None, 3], num_elements=3)
    e1 = gen_list_ops.tensor_list_get_item(
        l, 1, element_shape=[2, 3], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e1), np.zeros((2, 3)))

  def testGetUninitializedTensorWithInvalidElementShapeFails(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "Trying to read an uninitialized tensor but "
        "element_shape is not fully defined"):
      e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      self.evaluate(e0)

    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[None, 2], num_elements=3)

    # In eager mode the shape mismatch is caught in the TensorListGetItem
    # kernel which raises an InvalidArgumentError.
    # In graph mode the shape mismatch is caught in the C++ shape inference
    # which raises a ValueError.
    if context.executing_eagerly():
      error_type = errors.InvalidArgumentError
    else:
      error_type = ValueError
    with self.assertRaisesRegexp(error_type, r"shapes"):
      e0 = gen_list_ops.tensor_list_get_item(
          l, 0, element_dtype=dtypes.float32, element_shape=[1, 3])
      self.evaluate(e0)

  @test_util.run_deprecated_v1
  @test_util.enable_control_flow_v2
  def testSkipEagerSetItemIndexOutOfBounds(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[])
    e0 = constant_op.constant(5.)
    l = list_ops.tensor_list_set_item(
        l, 0, 2. * e0, resize_if_index_out_of_bounds=True)
    l = list_ops.tensor_list_set_item(
        l, 1, 1., resize_if_index_out_of_bounds=True)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    grad = gradients_impl.gradients(t, e0)[0]
    self.assertAllEqual(self.evaluate(grad), 2.)

  @test_util.run_deprecated_v1
  def testSetOnEmptyListWithMaxNumElementsFails(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[], max_num_elements=3)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "Trying to modify element 0 in a list with 0 elements."):
      l = list_ops.tensor_list_set_item(l, 0, 1.)
      self.evaluate(l)

  def testUnknownShape(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=None)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([1.0, 2.0]))
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e), [1.0, 2.0])
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e), 1.0)

  @test_util.run_gpu_only
  def testCPUGPUCopy(self):
    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(t, element_shape=[])
    with context.device("gpu:0"):
      l_gpu = array_ops.identity(l)
      self.assertAllEqual(
          self.evaluate(
              list_ops.tensor_list_pop_back(
                  l_gpu, element_dtype=dtypes.float32)[1]), 2.0)
    l_cpu = array_ops.identity(l_gpu)
    self.assertAllEqual(
        self.evaluate(
            list_ops.tensor_list_pop_back(
                l_cpu, element_dtype=dtypes.float32)[1]), 2.0)

  @test_util.run_gpu_only
  def testCPUGPUCopyNested(self):
    t = constant_op.constant([1.0, 2.0])
    child_l = list_ops.tensor_list_from_tensor(t, element_shape=[])
    l = list_ops.empty_tensor_list(
        element_shape=constant_op.constant([], dtype=dtypes.int32),
        element_dtype=dtypes.variant)
    l = list_ops.tensor_list_push_back(l, child_l)
    with context.device("gpu:0"):
      l_gpu = array_ops.identity(l)
      _, child_l_gpu = list_ops.tensor_list_pop_back(
          l_gpu, element_dtype=dtypes.variant)
      self.assertAllEqual(
          self.evaluate(
              list_ops.tensor_list_pop_back(
                  child_l_gpu, element_dtype=dtypes.float32)[1]), 2.0)
    l_cpu = array_ops.identity(l_gpu)
    _, child_l_cpu = list_ops.tensor_list_pop_back(
        l_cpu, element_dtype=dtypes.variant)
    self.assertAllEqual(
        self.evaluate(
            list_ops.tensor_list_pop_back(
                child_l_cpu, element_dtype=dtypes.float32)[1]), 2.0)

  def testGraphStack(self):
    with self.cached_session():
      tl = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([1], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      tl = list_ops.tensor_list_push_back(tl, [1])
      self.assertAllEqual(
          self.evaluate(
              list_ops.tensor_list_stack(tl, element_dtype=dtypes.int32)),
          [[1]])

  def testSkipEagerStackInLoop(self):
    with self.cached_session():
      t1 = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      i = constant_op.constant(0, dtype=dtypes.int32)

      def body(i, t1):
        t1 = list_ops.tensor_list_push_back(t1, i)
        i += 1
        return i, t1

      i, t1 = control_flow_ops.while_loop(lambda i, t1: math_ops.less(i, 4),
                                          body, [i, t1])
      s1 = list_ops.tensor_list_stack(t1, element_dtype=dtypes.int32)
      self.assertAllEqual(self.evaluate(s1), [0, 1, 2, 3])

  def testSkipEagerStackSwitchDtype(self):
    with self.cached_session():
      list_ = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      m = constant_op.constant([1, 2, 3], dtype=dtypes.float32)

      def body(list_, m):
        list_ = control_flow_ops.cond(
            math_ops.equal(list_ops.tensor_list_length(list_), 0),
            lambda: list_ops.empty_tensor_list(m.shape, m.dtype), lambda: list_)
        list_ = list_ops.tensor_list_push_back(list_, m)
        return list_, m

      for _ in range(2):
        list_, m = body(list_, m)

      s1 = list_ops.tensor_list_stack(list_, element_dtype=dtypes.float32)
      np_s1 = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
      self.assertAllEqual(self.evaluate(s1), np_s1)

  def testSkipEagerStackInLoopSwitchDtype(self):
    with self.cached_session():
      t1 = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      i = constant_op.constant(0, dtype=dtypes.float32)
      m = constant_op.constant([1, 2, 3], dtype=dtypes.float32)

      def body(i, m, t1):
        t1 = control_flow_ops.cond(
            math_ops.equal(list_ops.tensor_list_length(t1), 0),
            lambda: list_ops.empty_tensor_list(m.shape, m.dtype), lambda: t1)

        t1 = list_ops.tensor_list_push_back(t1, m * i)
        i += 1.0
        return i, m, t1

      i, m, t1 = control_flow_ops.while_loop(
          lambda i, m, t1: math_ops.less(i, 4), body, [i, m, t1])
      s1 = list_ops.tensor_list_stack(t1, element_dtype=dtypes.float32)
      np_s1 = np.vstack([np.arange(1, 4) * i for i in range(4)])
      self.assertAllEqual(self.evaluate(s1), np_s1)

  def testSerialize(self):
    worker = test_util.create_local_cluster(num_workers=1, num_ps=1)[0][0]
    with ops.Graph().as_default(), session.Session(target=worker.target):
      with ops.device("/job:worker"):
        t = constant_op.constant([[1.0], [2.0]])
        l = list_ops.tensor_list_from_tensor(t, element_shape=[1])
      with ops.device("/job:ps"):
        l_ps = array_ops.identity(l)
        l_ps, e = list_ops.tensor_list_pop_back(
            l_ps, element_dtype=dtypes.float32)
      with ops.device("/job:worker"):
        worker_e = array_ops.identity(e)
      self.assertAllEqual(self.evaluate(worker_e), [2.0])

  def testSerializeListWithInvalidTensors(self):
    worker = test_util.create_local_cluster(num_workers=1, num_ps=1)[0][0]
    with ops.Graph().as_default(), session.Session(target=worker.target):
      with ops.device("/job:worker"):
        l = list_ops.tensor_list_reserve(
            element_dtype=dtypes.float32, element_shape=[], num_elements=2)
        l = list_ops.tensor_list_set_item(l, 0, 1.)
      with ops.device("/job:ps"):
        l_ps = array_ops.identity(l)
        l_ps = list_ops.tensor_list_set_item(l_ps, 1, 2.)
        t = list_ops.tensor_list_stack(l_ps, element_dtype=dtypes.float32)
      with ops.device("/job:worker"):
        worker_t = array_ops.identity(t)
      self.assertAllEqual(self.evaluate(worker_t), [1.0, 2.0])

  def testSerializeListWithUnknownRank(self):
    worker = test_util.create_local_cluster(num_workers=1, num_ps=1)[0][0]
    with ops.Graph().as_default(), session.Session(target=worker.target):
      with ops.device("/job:worker"):
        t = constant_op.constant([[1.0], [2.0]])
        l = list_ops.tensor_list_from_tensor(t, element_shape=None)
      with ops.device("/job:ps"):
        l_ps = array_ops.identity(l)
        element_shape = list_ops.tensor_list_element_shape(
            l_ps, shape_type=dtypes.int32)
      with ops.device("/job:worker"):
        element_shape = array_ops.identity(element_shape)
      self.assertEqual(self.evaluate(element_shape), -1)

  def testSerializeListWithMaxNumElements(self):
    worker = test_util.create_local_cluster(num_workers=1, num_ps=1)[0][0]
    with ops.Graph().as_default(), session.Session(target=worker.target):
      with ops.device("/job:worker"):
        l = list_ops.empty_tensor_list(
            element_shape=None,
            element_dtype=dtypes.float32,
            max_num_elements=2)
        l = list_ops.tensor_list_push_back(l, 1.)
      with ops.device("/job:ps"):
        l_ps = array_ops.identity(l)
        l_ps = list_ops.tensor_list_push_back(l_ps, 2.)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "Tried to push item into a full list"):
        with ops.device("/job:worker"):
          l_worker = array_ops.identity(l_ps)
          l_worker = list_ops.tensor_list_push_back(l_worker, 3.0)
          self.evaluate(l_worker)

  def testPushPopGradients(self):
    with backprop.GradientTape() as tape:
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32, element_shape=[])
      c = constant_op.constant(1.0)
      tape.watch(c)
      l = list_ops.tensor_list_push_back(l, c)
      l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      e = 2 * e
    self.assertAllEqual(self.evaluate(tape.gradient(e, [c])[0]), 2.0)

  def testStackFromTensorGradients(self):
    with backprop.GradientTape() as tape:
      c = constant_op.constant([1.0, 2.0])
      tape.watch(c)
      l = list_ops.tensor_list_from_tensor(c, element_shape=[])
      c2 = list_ops.tensor_list_stack(
          l, element_dtype=dtypes.float32, num_elements=2)
      result = c2 * 2.0
    grad = tape.gradient(result, [c])[0]
    self.assertAllEqual(self.evaluate(grad), [2.0, 2.0])

  def testGetSetGradients(self):
    with backprop.GradientTape() as tape:
      c = constant_op.constant([1.0, 2.0])
      tape.watch(c)
      l = list_ops.tensor_list_from_tensor(c, element_shape=[])
      c2 = constant_op.constant(3.0)
      tape.watch(c2)
      l = list_ops.tensor_list_set_item(l, 0, c2)
      e = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      ee = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
      y = e * e + ee * ee
    grad_c, grad_c2 = tape.gradient(y, [c, c2])
    self.assertAllEqual(self.evaluate(grad_c), [0.0, 4.0])
    self.assertAllEqual(self.evaluate(grad_c2), 6.0)

  @test_util.run_deprecated_v1
  def testSetOutOfBounds(self):
    c = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(c, element_shape=[])
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(list_ops.tensor_list_set_item(l, 20, 3.0))

  @test_util.run_deprecated_v1
  def testSkipEagerSetItemWithMismatchedShapeFails(self):
    with self.cached_session() as sess:
      ph = array_ops.placeholder(dtypes.float32)
      c = constant_op.constant([1.0, 2.0])
      l = list_ops.tensor_list_from_tensor(c, element_shape=[])
      # Set a placeholder with unknown shape to satisfy the shape inference
      # at graph building time.
      l = list_ops.tensor_list_set_item(l, 0, ph)
      l_0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "incompatible shape"):
        sess.run(l_0, {ph: [3.0]})

  def testResourceVariableScatterGather(self):
    c = constant_op.constant([1.0, 2.0], dtype=dtypes.float32)
    l = list_ops.tensor_list_from_tensor(c, element_shape=[])
    v = vs.get_variable("var", initializer=[l] * 10, use_resource=True)
    v_r_0_stacked = list_ops.tensor_list_stack(v[0], dtypes.float32)
    self.evaluate(v.initializer)
    self.assertAllEqual([1.0, 2.0], self.evaluate(v_r_0_stacked))
    v_r_sparse_stacked = list_ops.tensor_list_stack(
        v.sparse_read(0), dtypes.float32)
    self.assertAllEqual([1.0, 2.0], self.evaluate(v_r_sparse_stacked))
    l_new_0 = list_ops.tensor_list_from_tensor([3.0, 4.0], element_shape=[])
    l_new_1 = list_ops.tensor_list_from_tensor([5.0, 6.0], element_shape=[])
    updated_v = state_ops.scatter_update(v, [3, 5], [l_new_0, l_new_1])
    updated_v_elems = array_ops.unstack(updated_v)
    updated_v_stacked = [
        list_ops.tensor_list_stack(el, dtypes.float32) for el in updated_v_elems
    ]
    expected = ([[1.0, 2.0]] * 3 + [[3.0, 4.0], [1.0, 2.0], [5.0, 6.0]] +
                [[1.0, 2.0]] * 4)
    self.assertAllEqual(self.evaluate(updated_v_stacked), expected)

  @test_util.run_deprecated_v1
  def testConcat(self):
    c = constant_op.constant([1.0, 2.0], dtype=dtypes.float32)
    l0 = list_ops.tensor_list_from_tensor(c, element_shape=[])
    l1 = list_ops.tensor_list_from_tensor([-1.0], element_shape=[])
    l_batch_0 = array_ops.stack([l0, l1])
    l_batch_1 = array_ops.stack([l1, l0])

    l_concat_01 = list_ops.tensor_list_concat_lists(
        l_batch_0, l_batch_1, element_dtype=dtypes.float32)
    l_concat_10 = list_ops.tensor_list_concat_lists(
        l_batch_1, l_batch_0, element_dtype=dtypes.float32)
    l_concat_00 = list_ops.tensor_list_concat_lists(
        l_batch_0, l_batch_0, element_dtype=dtypes.float32)
    l_concat_11 = list_ops.tensor_list_concat_lists(
        l_batch_1, l_batch_1, element_dtype=dtypes.float32)

    expected_0 = [[1.0, 2.0], [-1.0]]
    expected_1 = [[-1.0], [1.0, 2.0]]
    expected_00 = [[1.0, 2.0, 1.0, 2.0], [-1.0, -1.0]]
    expected_01 = [[1.0, 2.0, -1.0], [-1.0, 1.0, 2.0]]
    expected_10 = [[-1.0, 1.0, 2.0], [1.0, 2.0, -1.0]]
    expected_11 = [[-1.0, -1.0], [1.0, 2.0, 1.0, 2.0]]

    for i, (concat, expected) in enumerate(zip(
        [l_batch_0, l_batch_1,
         l_concat_00, l_concat_01, l_concat_10, l_concat_11],
        [expected_0, expected_1,
         expected_00, expected_01, expected_10, expected_11])):
      splitted = array_ops.unstack(concat)
      splitted_stacked_ret = self.evaluate(
          (list_ops.tensor_list_stack(splitted[0], dtypes.float32),
           list_ops.tensor_list_stack(splitted[1], dtypes.float32)))
      print("Test concat %d: %s, %s, %s, %s"
            % (i, expected[0], splitted_stacked_ret[0],
               expected[1], splitted_stacked_ret[1]))
      self.assertAllClose(expected[0], splitted_stacked_ret[0])
      self.assertAllClose(expected[1], splitted_stacked_ret[1])

    # Concatenating mismatched shapes fails.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      self.evaluate(
          list_ops.tensor_list_concat_lists(
              l_batch_0,
              list_ops.empty_tensor_list([], dtypes.float32),
              element_dtype=dtypes.float32))

    if context.executing_eagerly():
      expected_error = (
          errors.InvalidArgumentError,
          "element shapes are not identical at index 0")
    else:
      expected_error = (ValueError, "Shapes must be equal rank")
    with self.assertRaisesRegexp(*expected_error):
      l_batch_of_vec_tls = array_ops.stack(
          [list_ops.tensor_list_from_tensor([[1.0]], element_shape=[1])] * 2)
      self.evaluate(
          list_ops.tensor_list_concat_lists(l_batch_0, l_batch_of_vec_tls,
                                            element_dtype=dtypes.float32))

    if context.executing_eagerly():
      expected_error = (errors.InvalidArgumentError,
                        r"input_b\[0\].dtype != element_dtype.")
    else:
      expected_error = (ValueError, "input_b.type != element_dtype")
    with self.assertRaisesRegexp(*expected_error):
      l_batch_of_int_tls = array_ops.stack(
          [list_ops.tensor_list_from_tensor([1], element_shape=[])] * 2)
      self.evaluate(
          list_ops.tensor_list_concat_lists(l_batch_0, l_batch_of_int_tls,
                                            element_dtype=dtypes.float32))

  @test_util.run_deprecated_v1
  def testPushBackBatch(self):
    c = constant_op.constant([1.0, 2.0], dtype=dtypes.float32)
    l0 = list_ops.tensor_list_from_tensor(c, element_shape=[])
    l1 = list_ops.tensor_list_from_tensor([-1.0], element_shape=[])
    l_batch = array_ops.stack([l0, l1])
    l_push = list_ops.tensor_list_push_back_batch(l_batch, [3.0, 4.0])
    l_unstack = array_ops.unstack(l_push)
    l0_ret = list_ops.tensor_list_stack(l_unstack[0], dtypes.float32)
    l1_ret = list_ops.tensor_list_stack(l_unstack[1], dtypes.float32)
    self.assertAllClose([1.0, 2.0, 3.0], self.evaluate(l0_ret))
    self.assertAllClose([-1.0, 4.0], self.evaluate(l1_ret))

    with ops.control_dependencies([l_push]):
      l_unstack_orig = array_ops.unstack(l_batch)
      l0_orig_ret = list_ops.tensor_list_stack(l_unstack_orig[0],
                                               dtypes.float32)
      l1_orig_ret = list_ops.tensor_list_stack(l_unstack_orig[1],
                                               dtypes.float32)

    # Check that without aliasing, push_back_batch still works; and
    # that it doesn't modify the input.
    l0_r_v, l1_r_v, l0_orig_v, l1_orig_v = self.evaluate(
        (l0_ret, l1_ret, l0_orig_ret, l1_orig_ret))
    self.assertAllClose([1.0, 2.0, 3.0], l0_r_v)
    self.assertAllClose([-1.0, 4.0], l1_r_v)
    self.assertAllClose([1.0, 2.0], l0_orig_v)
    self.assertAllClose([-1.0], l1_orig_v)

    # Pushing back mismatched shapes fails.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      self.evaluate(list_ops.tensor_list_push_back_batch(l_batch, []))

    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "incompatible shape to a list at index 0"):
      self.evaluate(
          list_ops.tensor_list_push_back_batch(l_batch, [[3.0], [4.0]]))

    if context.executing_eagerly():
      expected_error = (errors.InvalidArgumentError, "Invalid data type")
    else:
      expected_error = (ValueError, "wrong element dtype")
    with self.assertRaisesRegexp(*expected_error):
      self.evaluate(list_ops.tensor_list_push_back_batch(l_batch, [3, 4]))

  def testZerosLike(self):
    for dtype in (dtypes.uint8, dtypes.uint16, dtypes.int8, dtypes.int16,
                  dtypes.int32, dtypes.int64, dtypes.float16, dtypes.float32,
                  dtypes.float64, dtypes.complex64, dtypes.complex128,
                  dtypes.bool):
      l_empty = list_ops.empty_tensor_list(
          element_dtype=dtype, element_shape=[])
      l_empty_zeros = array_ops.zeros_like(l_empty)
      t_empty_zeros = list_ops.tensor_list_stack(
          l_empty_zeros, element_dtype=dtype)

      l_full = list_ops.tensor_list_push_back(l_empty,
                                              math_ops.cast(0, dtype=dtype))
      l_full = list_ops.tensor_list_push_back(l_full,
                                              math_ops.cast(1, dtype=dtype))
      l_full_zeros = array_ops.zeros_like(l_full)
      t_full_zeros = list_ops.tensor_list_stack(
          l_full_zeros, element_dtype=dtype)

      self.assertAllEqual(self.evaluate(t_empty_zeros), [])
      self.assertAllEqual(
          self.evaluate(t_full_zeros), np.zeros(
              (2,), dtype=dtype.as_numpy_dtype))

  def testZerosLikeNested(self):
    for dtype in (dtypes.uint8, dtypes.uint16, dtypes.int8, dtypes.int16,
                  dtypes.int32, dtypes.int64, dtypes.float16, dtypes.float32,
                  dtypes.float64, dtypes.complex64, dtypes.complex128,
                  dtypes.bool):
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.variant, element_shape=[])

      sub_l = list_ops.empty_tensor_list(element_dtype=dtype, element_shape=[])
      l = list_ops.tensor_list_push_back(l, sub_l)
      sub_l = list_ops.tensor_list_push_back(sub_l, math_ops.cast(
          1, dtype=dtype))
      l = list_ops.tensor_list_push_back(l, sub_l)
      sub_l = list_ops.tensor_list_push_back(sub_l, math_ops.cast(
          2, dtype=dtype))
      l = list_ops.tensor_list_push_back(l, sub_l)

      # l : [[],
      #      [1],
      #      [1, 2]]
      #
      # l_zeros : [[],
      #            [0],
      #            [0, 0]]
      l_zeros = array_ops.zeros_like(l)

      outputs = []
      for _ in range(3):
        l_zeros, out = list_ops.tensor_list_pop_back(
            l_zeros, element_dtype=dtypes.variant)
        outputs.append(list_ops.tensor_list_stack(out, element_dtype=dtype))

      # Note: `outputs` contains popped values so the order is reversed.
      self.assertAllEqual(self.evaluate(outputs[2]), [])
      self.assertAllEqual(
          self.evaluate(outputs[1]), np.zeros((1,), dtype=dtype.as_numpy_dtype))
      self.assertAllEqual(
          self.evaluate(outputs[0]), np.zeros((2,), dtype=dtype.as_numpy_dtype))

  def testElementShape(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=None)
    shape = list_ops.tensor_list_element_shape(l, shape_type=dtypes.int32)
    self.assertEqual(self.evaluate(shape), -1)

  def testZerosLikeUninitialized(self):
    l0 = list_ops.tensor_list_reserve([], 3, element_dtype=dtypes.float32)
    l1 = list_ops.tensor_list_set_item(l0, 0, 1.)  # [1., _, _]
    zeros_1 = array_ops.zeros_like(l1)  # [0., _, _]
    l2 = list_ops.tensor_list_set_item(l1, 2, 2.)  # [1., _, 2.]
    zeros_2 = array_ops.zeros_like(l2)  # [0., _, 0.]

    # Gather indices with zeros in `zeros_1`.
    res_1 = list_ops.tensor_list_gather(
        zeros_1, [0], element_dtype=dtypes.float32)
    # Gather indices with zeros in `zeros_2`.
    res_2 = list_ops.tensor_list_gather(
        zeros_2, [0, 2], element_dtype=dtypes.float32)

    self.assertAllEqual(self.evaluate(res_1), [0.])
    self.assertAllEqual(self.evaluate(res_2), [0., 0.])

  @test_util.run_deprecated_v1
  def testSkipEagerTensorListGetItemGradAggregation(self):
    l = list_ops.tensor_list_reserve(
        element_shape=[], num_elements=1, element_dtype=dtypes.float32)
    x = constant_op.constant(1.0)
    l = list_ops.tensor_list_set_item(l, 0, x)
    l_read1 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
    l_read2 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
    grad = gradients_impl.gradients([l_read1, l_read2], [x])
    with self.cached_session() as sess:
      self.assertSequenceEqual(self.evaluate(grad), [2.])

  @test_util.run_deprecated_v1
  def testSkipEagerBuildElementShape(self):
    fn = list_ops._build_element_shape
    # Unknown shape -> -1.
    self.assertEqual(fn(None), -1)
    self.assertEqual(fn(tensor_shape.unknown_shape()), -1)
    # Scalar shape -> [] with type int32.
    self.assertEqual(fn([]).dtype, dtypes.int32)
    self.assertEqual(fn(tensor_shape.TensorShape([])).dtype, dtypes.int32)
    self.assertAllEqual(self.evaluate(fn([])), np.array([], np.int32))
    self.assertAllEqual(
        self.evaluate(fn(tensor_shape.TensorShape([]))), np.array([], np.int32))
    # Tensor -> Tensor
    shape = constant_op.constant(1)
    self.assertIs(fn(shape), shape)
    # Shape with unknown dims -> shape list with -1's.
    shape = [None, 5]
    self.assertAllEqual(fn(shape), [-1, 5])
    self.assertAllEqual(fn(tensor_shape.TensorShape(shape)), [-1, 5])
    # Shape with unknown dims and tensor dims -> shape list with -1's and tensor
    # dims.
    t = array_ops.placeholder(dtypes.int32)
    shape = [None, 5, t]
    result = fn(shape)
    self.assertAllEqual(result[:2], [-1, 5])
    self.assertIs(result[2], t)

  def testAddN(self):
    l1 = list_ops.tensor_list_from_tensor([1.0, 2.0], element_shape=[])
    l2 = list_ops.tensor_list_from_tensor([3.0, 4.0], element_shape=[])
    l3 = list_ops.tensor_list_from_tensor([5.0, 6.0], element_shape=[])
    result = math_ops.add_n((l1, l2, l3))
    result_t = list_ops.tensor_list_stack(result, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(result_t), [9., 12.])

  def testAddNNestedList(self):
    l1 = list_ops.tensor_list_from_tensor([1.0, 2.0], element_shape=[])
    l2 = list_ops.tensor_list_from_tensor([3.0, 4.0], element_shape=[])
    l3 = list_ops.tensor_list_from_tensor([5.0, 6.0], element_shape=[])
    l4 = list_ops.tensor_list_from_tensor([7.0, 8.0], element_shape=[])
    a = list_ops.empty_tensor_list(
        element_dtype=dtypes.variant, element_shape=[])
    a = list_ops.tensor_list_push_back(a, l1)
    a = list_ops.tensor_list_push_back(a, l2)
    b = list_ops.empty_tensor_list(
        element_dtype=dtypes.variant, element_shape=[])
    b = list_ops.tensor_list_push_back(b, l3)
    b = list_ops.tensor_list_push_back(b, l4)
    result = math_ops.add_n((a, b))
    result_0 = list_ops.tensor_list_stack(
        list_ops.tensor_list_get_item(result, 0, element_dtype=dtypes.variant),
        element_dtype=dtypes.float32)
    result_1 = list_ops.tensor_list_stack(
        list_ops.tensor_list_get_item(result, 1, element_dtype=dtypes.variant),
        element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(result_0), [6., 8.])
    self.assertAllEqual(self.evaluate(result_1), [10., 12.])

  def testAddTensorListsFailsIfLeadingDimsMismatch(self):
    l1 = list_ops.tensor_list_reserve(
        element_shape=[], element_dtype=dtypes.float32, num_elements=2)
    l2 = list_ops.tensor_list_reserve(
        element_shape=[], element_dtype=dtypes.float32, num_elements=3)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "Trying to add two lists of tensors with different lengths"):
      l = math_ops.add_n([l1, l2])
      self.evaluate(list_ops.tensor_list_stack(l, element_dtype=dtypes.float32))

  @test_util.run_v1_only("Uses placeholders")
  def testSkipEagerAddTensorListsFailsIfElementShapesMismatch(self):
    with self.cached_session() as sess:
      # Use placeholders instead of constant values for shapes to prevent TF's
      # shape inference from catching this early.
      l1_element_shape = array_ops.placeholder(dtype=dtypes.int32)
      l2_element_shape = array_ops.placeholder(dtype=dtypes.int32)
      l1 = list_ops.tensor_list_reserve(
          element_shape=l1_element_shape,
          element_dtype=dtypes.float32,
          num_elements=3)
      l2 = list_ops.tensor_list_reserve(
          element_shape=l2_element_shape,
          element_dtype=dtypes.float32,
          num_elements=3)
      l = math_ops.add_n([l1, l2])
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "Trying to add two lists of tensors with incompatible element shapes"
      ):
        sess.run(
            list_ops.tensor_list_stack(l, element_dtype=dtypes.float32), {
                l1_element_shape: [],
                l2_element_shape: [2]
            })

  @test_util.run_deprecated_v1
  def testSkipEagerConcatShapeInference(self):

    def BuildTensor(element_shape):
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32, element_shape=element_shape)
      return list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)

    self.assertIsNone(BuildTensor(None).shape.rank)
    self.assertAllEqual(BuildTensor([None, 2, 3]).shape.as_list(), [None, 2, 3])
    self.assertAllEqual(
        BuildTensor([None, 2, None]).shape.as_list(), [None, 2, None])
    self.assertAllEqual(BuildTensor([1, 2, 3]).shape.as_list(), [None, 2, 3])

  def testConcatWithFullyDefinedElementShape(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[2, 2])
    l = list_ops.tensor_list_push_back(l, [[0., 1.], [2., 3.]])
    l = list_ops.tensor_list_push_back(l, [[4., 5.], [6., 7.]])
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual(
        self.evaluate(t), [[0., 1.], [2., 3.], [4., 5.], [6., 7.]])

  def testConcatWithNonFullyDefinedElementShape(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[None, 2])
    l = list_ops.tensor_list_push_back(l, [[0., 1.]])
    l = list_ops.tensor_list_push_back(l, [[2., 3.], [4., 5.]])
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[0., 1.], [2., 3.], [4., 5.]])

  def testConcatWithMismatchingTensorShapesFails(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=None)
    l = list_ops.tensor_list_push_back(l, [[0., 1.]])
    l = list_ops.tensor_list_push_back(l, [[2.], [4.]])
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError, r"Incompatible shapes during merge: "
        r"\[2\] vs. \[1\]"):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatEmptyListWithFullyDefinedElementShape(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[5, 2])
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t).shape, (0, 2))
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[None, 2])
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t).shape, (0, 2))

  def testConcatEmptyListWithUnknownElementShapeFails(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=None)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "All except the first dimension must be fully"
        " defined when concating an empty tensor list"):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatEmptyListWithPartiallyDefinedElementShapeFails(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[2, None])
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "All except the first dimension must be fully"
        " defined when concating an empty tensor list"):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatListWithScalarElementShapeFails(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=tensor_shape.TensorShape([]))
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "Concat requires elements to be at least vectors, "
        "found scalars instead"):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatListWithScalarElementsFails(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=None)
    l1 = list_ops.tensor_list_push_back(l, 1.)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError, "Concat saw a scalar shape at index 0"
        " but requires at least vectors"):
      t = list_ops.tensor_list_concat(l1, element_dtype=dtypes.float32)
      self.evaluate(t)
    l1 = list_ops.tensor_list_push_back(l, [1.])
    l1 = list_ops.tensor_list_push_back(l1, 2.)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError, "Concat saw a scalar shape at index 1"
        " but requires at least vectors"):
      t = list_ops.tensor_list_concat(l1, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatWithUninitializedTensorsUseListElementShape(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[2, 3], num_elements=3)
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual(np.zeros((6, 3)), t)

  def testConcatWithUninitializedTensorsUseProvidedElementShape(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    t = list_ops.tensor_list_concat(
        l, element_dtype=dtypes.float32, element_shape=(2, 3))
    self.assertAllEqual(np.zeros((6, 3)), t)

  def testConcatWithUninitializedTensorsUseProvidedElementShapeAndLengths(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    t, _ = gen_list_ops.tensor_list_concat_v2(
        l,
        element_dtype=dtypes.float32,
        element_shape=list_ops._build_element_shape((None, 3)),
        leading_dims=[2, 3, 5])
    self.assertAllEqual(np.zeros((10, 3)), t)
    l = list_ops.tensor_list_set_item(l, 1, [[2., 3.], [4., 5.], [6., 7.]])
    t, _ = gen_list_ops.tensor_list_concat_v2(
        l,
        element_dtype=dtypes.float32,
        element_shape=list_ops._build_element_shape((None, 2)),
        leading_dims=[2, 3, 4])
    self.assertAllEqual([[0., 0.], [0., 0.], [2., 3.], [4., 5.], [6., 7.],
                         [0., 0.], [0., 0.], [0., 0.], [0., 0.]], t)

  def testConcatWithUninitializedTensorsInferShapeFromElements(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    l = list_ops.tensor_list_set_item(l, 1, [[2., 3.], [4., 5.], [6., 7.]])
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual([[0., 0.], [0., 0.], [0., 0.], [2., 3.], [4., 5.],
                         [6., 7.], [0., 0.], [0., 0.], [0., 0.]], t)

  def testConcatWithUninitializedTensorsFailsIfNoElementShape(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r"Trying to concat list with only uninitialized tensors "
        r"but element_shape_except_first_dim_ is not fully defined"):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatWithUninitializedTensorsFailsIfNoInputLengths(self):
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[None, 3], num_elements=3)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r"List contains uninitialized tensor at index 0"
        r" but leading_dims has only 0 elements."):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testEvenSplit(self):

    def RunTest(input_tensor, lengths, expected_stacked_output):
      l = list_ops.tensor_list_split(
          input_tensor, element_shape=None, lengths=lengths)
      self.assertAllEqual(
          list_ops.tensor_list_stack(l, element_dtype=dtypes.float32),
          expected_stacked_output)

    RunTest([1., 2., 3.], [1, 1, 1], [[1.], [2.], [3.]])
    RunTest([1., 2., 3., 4.], [2, 2], [[1., 2.], [3., 4.]])
    RunTest([[1., 2.], [3., 4.]], [1, 1], [[[1., 2.]], [[3., 4.]]])

  def testUnevenSplit(self):
    l = list_ops.tensor_list_split([1., 2., 3., 4., 5],
                                   element_shape=None,
                                   lengths=[3, 2])
    self.assertAllEqual(list_ops.tensor_list_length(l), 2)
    self.assertAllEqual(
        list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32),
        [1., 2., 3.])
    self.assertAllEqual(
        list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32),
        [4., 5.])

  @test_util.run_deprecated_v1
  def testSkipEagerSplitWithInvalidTensorShapeFails(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtype=dtypes.float32)
      l = list_ops.tensor_list_split(tensor, element_shape=None, lengths=[1])
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r"Tensor must be at least a vector, but saw shape: \[\]"):
        l.eval({tensor: 1})

  @test_util.run_deprecated_v1
  def testSkipEagerSplitWithInvalidLengthsShapeFails(self):
    with self.cached_session():
      lengths = array_ops.placeholder(dtype=dtypes.int64)
      l = list_ops.tensor_list_split([1., 2.],
                                     element_shape=None,
                                     lengths=lengths)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r"Expected lengths to be a vector, received shape: \[\]"):
        l.eval({lengths: 1})

  def testSplitWithInvalidLengthsFails(self):
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 r"Invalid value in lengths: -1"):
      l = list_ops.tensor_list_split([1., 2.],
                                     element_shape=None,
                                     lengths=[1, -1])
      self.evaluate(l)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r"Attempting to slice \[0, 3\] from tensor with length 2"):
      l = list_ops.tensor_list_split([1., 2.], element_shape=None, lengths=[3])
      self.evaluate(l)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r"Unused values in tensor. Length of tensor: 2 Values used: 1"):
      l = list_ops.tensor_list_split([1., 2.], element_shape=None, lengths=[1])
      self.evaluate(l)

  @test_util.run_deprecated_v1
  def testSkipEagerSplitWithScalarElementShapeFails(self):
    with self.assertRaisesRegexp(ValueError,
                                 r"Shapes must be equal rank, but are 1 and 0"):
      l = list_ops.tensor_list_split([1., 2.], element_shape=[], lengths=[1, 1])
    with self.cached_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r"TensorListSplit requires element_shape to be at least of rank 1, "
          r"but saw: \[\]"):
        element_shape = array_ops.placeholder(dtype=dtypes.int32)
        l = list_ops.tensor_list_split([1., 2.],
                                       element_shape=element_shape,
                                       lengths=[1, 1])
        l.eval({element_shape: []})

  def testEagerOnlySplitWithScalarElementShapeFails(self):
    if context.executing_eagerly():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r"TensorListSplit requires element_shape to be at least of rank 1, "
          r"but saw: \[\]"):
        list_ops.tensor_list_split([1., 2.], element_shape=[], lengths=[1, 1])

  @test_util.run_deprecated_v1
  def testSkipEagerSplitWithIncompatibleTensorShapeAndElementShapeFails(self):
    with self.assertRaisesRegexp(ValueError,
                                 r"Shapes must be equal rank, but are 2 and 1"):
      l = list_ops.tensor_list_split([[1.], [2.]],
                                     element_shape=[1],
                                     lengths=[1, 1])

    with self.cached_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r"tensor shape \[2,1\] is not compatible with element_shape \[1\]"):
        element_shape = array_ops.placeholder(dtype=dtypes.int32)
        l = list_ops.tensor_list_split([[1.], [2.]],
                                       element_shape=element_shape,
                                       lengths=[1, 1])
        l.eval({element_shape: [1]})

  def testEagerOnlySplitWithIncompatibleTensorShapeAndElementShapeFails(self):
    if context.executing_eagerly():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r"tensor shape \[2,1\] is not compatible with element_shape \[1\]"):
        list_ops.tensor_list_split([[1.], [2.]],
                                   element_shape=[1],
                                   lengths=[1, 1])

  def testResizeGrow(self):
    l = list_ops.tensor_list_from_tensor([1., 2.], element_shape=[])
    l = list_ops.tensor_list_resize(l, 4)
    self.assertEqual(self.evaluate(list_ops.tensor_list_length(l)), 4)
    self.assertEqual(
        self.evaluate(
            list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)),
        1.)
    self.assertEqual(
        self.evaluate(
            list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)),
        2.)

  def testResizeShrink(self):
    l = list_ops.tensor_list_from_tensor([1., 2., 3.], element_shape=[])
    l = list_ops.tensor_list_resize(l, 2)
    self.assertEqual(self.evaluate(list_ops.tensor_list_length(l)), 2)
    self.assertAllEqual(
        self.evaluate(
            list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)),
        [1., 2.])

  def testResizeWithInvalidSizeFails(self):
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "TensorListSlice expects size to be non-negative"):
      l = list_ops.tensor_list_from_tensor([1., 2., 3.], element_shape=[])
      l = list_ops.tensor_list_resize(l, -1)
      self.evaluate(l)

  @test_util.run_deprecated_v1
  @test_util.enable_control_flow_v2
  def testSkipEagerResizeGrad(self):
    t = constant_op.constant([1., 2., 3.])
    l = list_ops.tensor_list_from_tensor(t, element_shape=[])
    l = list_ops.tensor_list_set_item(
        l, 3, 4., resize_if_index_out_of_bounds=True)
    t1 = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    grad = gradients_impl.gradients(t1, t)[0]
    self.assertAllEqual(self.evaluate(grad), [1., 1., 1.])

  def testHandleDataAcrossFunctionCall(self):

    @def_function.function
    def func():
      t = constant_op.constant([1., 2., 3.])
      l = list_ops.tensor_list_from_tensor(t, element_shape=[])
      return l

    tensor_list = func()
    element = list_ops.tensor_list_get_item(
        tensor_list, 0, element_dtype=dtypes.float32)
    self.assertAllEqual(element.shape.as_list(), [])

  @test_util.run_gpu_only
  def testNestedListDevicetoDeviceCopy(self):
    if context.num_gpus() < 2:
      self.skipTest("Need at least 2 GPUs for this test, found %d" %
                    context.num_gpus())
    with ops.device("gpu:0"):
      t = constant_op.constant([1.0, 2.0, 3.0])
      inner_l = list_ops.tensor_list_from_tensor(t, element_shape=[])
      outer_l = list_ops.empty_tensor_list(
          element_dtype=dtypes.variant, element_shape=[])
      outer_l = list_ops.tensor_list_push_back(outer_l, inner_l)

    # Stress test.
    for _ in range(1024):
      with ops.device("gpu:1"):
        outer_l = array_ops.identity(outer_l)
      with ops.device("gpu:0"):
        outer_l = array_ops.identity(outer_l)

    with ops.device("gpu:1"):
      _, inner_l = list_ops.tensor_list_pop_back(
          outer_l, element_dtype=dtypes.variant)
      t = list_ops.tensor_list_stack(inner_l, element_dtype=dtypes.float32)
      self.assertAllEqual(t, [1.0, 2.0, 3.0])


if __name__ == "__main__":
  test.main()
