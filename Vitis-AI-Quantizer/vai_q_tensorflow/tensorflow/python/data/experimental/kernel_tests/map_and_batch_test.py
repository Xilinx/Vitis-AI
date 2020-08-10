# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.map_and_batch()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class MapAndBatchTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Default", None, None),
      ("SequentialCalls", 1, None),
      ("ParallelCalls", 2, None),
      ("ParallelBatches", None, 10),
  )
  def testMapAndBatch(self, num_parallel_calls, num_parallel_batches):
    """Test a dataset that maps a TF function across its input elements."""
    # The pipeline is TensorSliceDataset ->
    # RepeatDataset(count) -> MapAndBatchDataset(square_3, batch_size).
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    def dataset_fn(batch_size, count):
      dataset = dataset_ops.Dataset.from_tensor_slices(components).repeat(
          count).apply(
              batching.map_and_batch(
                  map_func=_map_fn,
                  batch_size=batch_size,
                  num_parallel_calls=num_parallel_calls,
                  num_parallel_batches=num_parallel_batches))
      return dataset

    # Batch of a finite input, where the batch_size divides the
    # total number of elements.
    dataset = dataset_fn(14, 28)
    get_next = self.getNext(dataset)
    self.assertEqual(
        [[None] + list(c.shape[1:]) for c in components],
        [shape.as_list()
         for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    num_batches = (28 * 7) // 14
    for i in range(num_batches):
      result = self.evaluate(get_next())
      for component, result_component in zip(components, result):
        for j in range(14):
          self.assertAllEqual(component[(i * 14 + j) % 7]**2,
                              result_component[j])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    # Batch of a finite input, where the batch_size does not
    # divide the total number of elements.
    get_next = self.getNext(dataset_fn(8, 14))

    # We expect (num_batches - 1) full-sized batches.
    num_batches = int(math.ceil((14 * 7) / 8))
    for i in range(num_batches - 1):
      result = self.evaluate(get_next())
      for component, result_component in zip(components, result):
        for j in range(8):
          self.assertAllEqual(component[(i * 8 + j) % 7]**2,
                              result_component[j])

    result = self.evaluate(get_next())
    for component, result_component in zip(components, result):
      for j in range((14 * 7) % 8):
        self.assertAllEqual(component[((num_batches - 1) * 8 + j) % 7]**2,
                            result_component[j])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    # Batch of an empty input should fail straight away.
    self.assertDatasetProduces(dataset_fn(8, 0), expected_output=[])

    # Empty batch should be an initialization time error.
    with self.assertRaises(errors.InvalidArgumentError):
      self.assertDatasetProduces(dataset_fn(0, 14), expected_output=[])

  @parameterized.named_parameters(
      ("Even", False),
      ("Uneven", True),
  )
  def testMapAndBatchPartialBatch(self, drop_remainder):
    dataset = (
        dataset_ops.Dataset.range(10).apply(
            batching.map_and_batch(
                lambda x: array_ops.reshape(x * x, [1]),
                batch_size=4,
                drop_remainder=drop_remainder)))

    if drop_remainder:
      self.assertEqual(
          [4, 1], dataset_ops.get_legacy_output_shapes(dataset).as_list())
    else:
      self.assertEqual(
          [None, 1], dataset_ops.get_legacy_output_shapes(dataset).as_list())
    expected_output = [[[0], [1], [4], [9]], [[16], [25], [36], [49]]]
    if not drop_remainder:
      expected_output.append([[64], [81]])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  def testMapAndBatchYieldsPartialBatch(self):
    dataset = (
        dataset_ops.Dataset.range(10).apply(
            batching.map_and_batch(lambda x: array_ops.reshape(x * x, [1]), 4)))

    self.assertEqual(
        [None, 1], dataset_ops.get_legacy_output_shapes(dataset).as_list())
    expected_output = [[[0], [1], [4], [9]], [[16], [25], [36], [49]],
                       [[64], [81]]]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  def testMapAndBatchParallelGetNext(self):
    dataset = dataset_ops.Dataset.range(50000).apply(
        batching.map_and_batch(lambda x: x, batch_size=100))

    if context.executing_eagerly():
      iterator = iter(dataset)
      get_next = iterator._next_internal  # pylint: disable=protected-access
    else:
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      get_next = iterator.get_next

    elements = []
    for _ in range(100):
      elements.append(get_next)

    for i in range(5):
      got = self.evaluate([element() for element in elements])
      got.sort(key=lambda x: x[0])
      expected = []
      for j in range(100):
        expected.append(range(i * 10000 + j * 100, i * 10000 + (j + 1) * 100))
      self.assertAllEqual(got, expected)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate([element() for element in elements])

  def testMapAndBatchParallelGetNextDropRemainder(self):
    dataset = dataset_ops.Dataset.range(49999).apply(
        batching.map_and_batch(
            lambda x: x, batch_size=100, drop_remainder=True))

    if context.executing_eagerly():
      iterator = iter(dataset)
      get_next = iterator._next_internal  # pylint: disable=protected-access
    else:
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      get_next = iterator.get_next

    elements = []
    for _ in range(100):
      elements.append(get_next)

    for i in range(4):
      got = self.evaluate([element() for element in elements])
      got.sort(key=lambda x: x[0])
      expected = []
      for j in range(100):
        expected.append(range(i * 10000 + j * 100, i * 10000 + (j + 1) * 100))
      self.assertAllEqual(got, expected)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate([element() for element in elements])

  def testMapAndBatchSparse(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    dataset = dataset_ops.Dataset.range(10).apply(
        batching.map_and_batch(_sparse, 5))

    self.assertDatasetProduces(
        dataset,
        expected_output=[
            sparse_tensor.SparseTensorValue(
                indices=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
                values=[i * 5, i * 5 + 1, i * 5 + 2, i * 5 + 3, i * 5 + 4],
                dense_shape=[5, 1]) for i in range(2)
        ])

  def testMapAndBatchFails(self):
    """Test a dataset that maps a TF function across its input elements."""

    with self.assertRaisesRegexp(errors.InvalidArgumentError, "oops"):
      dataset = dataset_ops.Dataset.from_tensors(
          array_ops.check_numerics(
              constant_op.constant(1.0) / constant_op.constant(0.0), "oops"))
      dataset = dataset.apply(batching.map_and_batch(lambda x: x, 14))
      get_next = self.getNext(dataset, requires_initialization=True)
      self.evaluate(get_next())

  def testMapAndBatchShapeMismatch(self):
    """Test a dataset that maps a TF function across its input elements."""

    def generator():
      yield [1]
      yield [2]
      yield [3]
      yield [[4, 5, 6]]

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int32)
    batch_size = 4
    dataset = dataset.apply(batching.map_and_batch(lambda x: x, batch_size))
    self.assertDatasetProduces(
        dataset,
        expected_error=(errors.InvalidArgumentError,
                        "number of elements does not match"))

  def testMapAndBatchImplicitDispose(self):
    # Tests whether a map and batch dataset will be cleaned up correctly when
    # the pipeline does not run it until exhaustion.
    # The pipeline is TensorSliceDataset -> RepeatDataset(1000) ->
    # MapAndBatchDataset(f=square_3, batch_size=100).
    components = (np.arange(1000),
                  np.array([[1, 2, 3]]) * np.arange(1000)[:, np.newaxis],
                  np.array(37.0) * np.arange(1000))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    dataset = dataset_ops.Dataset.from_tensor_slices(components).repeat(
        1000).apply(batching.map_and_batch(_map_fn, batch_size=100))
    dataset = dataset.prefetch(5)
    get_next = self.getNext(dataset)
    for _ in range(3):
      self.evaluate(get_next())

  @parameterized.named_parameters(
      ("1", 0),
      ("2", 5),
      ("3", 10),
      ("4", 90),
      ("5", 95),
      ("6", 99),
  )
  def testMapAndBatchMapError(self, threshold):

    def raising_py_fn(i):
      if i >= threshold:
        raise StopIteration()
      else:
        return i

    dataset = dataset_ops.Dataset.range(100).apply(
        batching.map_and_batch(
            lambda x: script_ops.py_func(raising_py_fn, [x], dtypes.int64),
            batch_size=10))

    get_next = self.getNext(dataset)
    for i in range(threshold // 10):
      self.assertAllEqual([i * 10 + j for j in range(10)],
                          self.evaluate(get_next()))
    for i in range(threshold // 10, 10):
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @parameterized.named_parameters(
      ("1", False, dtypes.bool),
      ("2", -42, dtypes.int8),
      ("3", -42, dtypes.int16),
      ("4", -42, dtypes.int32),
      ("5", -42, dtypes.int64),
      ("6", 42, dtypes.uint8),
      ("7", 42, dtypes.uint16),
      ("8", 42.0, dtypes.float16),
      ("9", 42.0, dtypes.float32),
      ("10", 42.0, dtypes.float64),
      ("11", b"hello", dtypes.string),
  )
  def testMapAndBatchTypes(self, element, dtype):

    def gen():
      yield element

    dataset = dataset_ops.Dataset.from_generator(gen, dtype).repeat(100).apply(
        batching.map_and_batch(lambda x: x, batch_size=10))

    get_next = self.getNext(dataset)
    for _ in range(10):
      self.assertAllEqual([element for _ in range(10)],
                          self.evaluate(get_next()))

  @parameterized.named_parameters(
      ("Identity", None, lambda x: x, None),
      ("Replicate", None, lambda x: (x, x), None),
      ("Swap", (None, None), lambda x, y: (y, x), None),
      ("Project", (None, None), lambda x, y: x, None),
  )
  def testShortCircuit(self, structure, map_fn, num_parallel_calls):
    dataset = self.structuredDataset(structure).repeat().apply(
        batching.map_and_batch(map_fn, batch_size=10))
    get_next = self.getNext(dataset)

    if isinstance(structure, tuple):
      expected = map_fn(
          *self.evaluate(self.structuredElement(structure, shape=[10])))
    else:
      expected = map_fn(
          self.evaluate(self.structuredElement(structure, shape=[10])))
    self.assertAllEqual(expected, self.evaluate(get_next()))

  def testShortCircuitCapturedInput(self):
    captured_t = variables.Variable(42)
    dataset = self.structuredDataset(None).repeat().apply(
        batching.map_and_batch(lambda x: captured_t, batch_size=10))
    self.evaluate(variables.global_variables_initializer())
    get_next = self.getNext(dataset, requires_initialization=True)
    self.assertAllEqual([42] * 10, self.evaluate(get_next()))

  def testMapAndBatchControlFlow(self):

    def map_fn(x):
      previous_control_flow_v2_value = control_flow_util.ENABLE_CONTROL_FLOW_V2
      control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
      return_value = control_flow_ops.cond(x < 50, lambda: x + 1, lambda: x * x)
      control_flow_util.ENABLE_CONTROL_FLOW_V2 = previous_control_flow_v2_value
      return return_value

    dataset = dataset_ops.Dataset.range(100).apply(
        batching.map_and_batch(map_fn, batch_size=10))
    get_next = self.getNext(dataset)
    for i in range(10):
      if i < 5:
        self.assertAllEqual([i * 10 + j + 1 for j in range(10)],
                            self.evaluate(get_next()))
      else:
        self.assertAllEqual(
            [((i * 10) + j) * ((i * 10) + j) for j in range(10)],
            self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())


if __name__ == "__main__":
  test.main()
