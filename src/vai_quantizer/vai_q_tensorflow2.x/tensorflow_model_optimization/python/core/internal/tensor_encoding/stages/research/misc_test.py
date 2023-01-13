# Copyright 2019, The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research import misc
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing import test_utils


if tf.executing_eagerly():
  tf.compat.v1.disable_eager_execution()


class SplitBySmallValueEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return misc.SplitBySmallValueEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([50], minval=-1.0, maxval=1.0)

  @property
  def is_lossless(self):
    """See base class."""
    return False

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self._assert_is_integer(
        data.encoded_x[misc.SplitBySmallValueEncodingStage.ENCODED_INDICES_KEY])

  def _assert_is_integer(self, indices):
    """Asserts that indices values are integers."""
    assert indices.dtype == np.int32

  @parameterized.parameters([tf.float32, tf.float64])
  def test_input_types(self, x_dtype):
    # Tests different input dtypes.
    x = tf.constant([1.0, 0.1, 0.01, 0.001, 0.0001], dtype=x_dtype)
    threshold = 0.05
    stage = misc.SplitBySmallValueEncodingStage(threshold=threshold)
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = test_utils.TestData(x, encoded_x, decoded_x)
    test_data = self.evaluate_test_data(test_data)

    self._assert_is_integer(test_data.encoded_x[
        misc.SplitBySmallValueEncodingStage.ENCODED_INDICES_KEY])

    # The numpy arrays must have the same dtype as the arrays from test_data.
    expected_encoded_values = np.array([1.0, 0.1], dtype=x.dtype.as_numpy_dtype)
    expected_encoded_indices = np.array([0, 1], dtype=np.int32)
    expected_decoded_x = np.array([1.0, 0.1, 0., 0., 0.],
                                  dtype=x_dtype.as_numpy_dtype)
    self.assertAllEqual(test_data.encoded_x[stage.ENCODED_VALUES_KEY],
                        expected_encoded_values)
    self.assertAllEqual(test_data.encoded_x[stage.ENCODED_INDICES_KEY],
                        expected_encoded_indices)
    self.assertAllEqual(test_data.decoded_x, expected_decoded_x)

  def test_all_zero_input_works(self):
    # Tests that encoding does not blow up with all-zero input. With all-zero
    # input, both of the encoded values will be empty arrays.
    stage = misc.SplitBySmallValueEncodingStage()
    test_data = self.run_one_to_many_encode_decode(stage,
                                                   lambda: tf.zeros([50]))

    self.assertAllEqual(np.zeros((50)).astype(np.float32), test_data.decoded_x)

  def test_all_below_threshold_works(self):
    # Tests that encoding does not blow up with all-below-threshold input. In
    # this case, both of the encoded values will be empty arrays.
    stage = misc.SplitBySmallValueEncodingStage(threshold=0.1)
    x = tf.random.uniform([50], minval=-0.01, maxval=0.01)
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = test_utils.TestData(x, encoded_x, decoded_x)
    test_data = self.evaluate_test_data(test_data)

    expected_encoded_indices = np.array([], dtype=np.int32).reshape([0])
    self.assertAllEqual(test_data.encoded_x[stage.ENCODED_VALUES_KEY], [])
    self.assertAllEqual(test_data.encoded_x[stage.ENCODED_INDICES_KEY],
                        expected_encoded_indices)
    self.assertAllEqual(test_data.decoded_x,
                        np.zeros([50], dtype=x.dtype.as_numpy_dtype))


class DifferenceBetweenIntegersEncodingStageTest(
    test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return misc.DifferenceBetweenIntegersEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([10], minval=0, maxval=10, dtype=tf.int64)

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllEqual(data.x, data.decoded_x)

  @parameterized.parameters(
      itertools.product([[1,], [2,], [10,]], [tf.int32, tf.int64]))
  def test_with_multiple_input_shapes(self, input_dims, dtype):

    def x_fn():
      return tf.random.uniform(input_dims, minval=0, maxval=10, dtype=dtype)

    test_data = self.run_one_to_many_encode_decode(
        self.default_encoding_stage(), x_fn)
    self.common_asserts_for_test_data(test_data)

  def test_empty_input_static(self):
    # Tests that the encoding works when the input shape is [0].
    x = []
    x = tf.convert_to_tensor(x, dtype=tf.int32)
    assert x.shape.as_list() == [0]

    stage = self.default_encoding_stage()
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)

    test_data = self.evaluate_test_data(
        test_utils.TestData(x, encoded_x, decoded_x))
    self.common_asserts_for_test_data(test_data)

  def test_empty_input_dynamic(self):
    # Tests that the encoding works when the input shape is [0], but not
    # statically known.
    y = tf.zeros((10,))
    indices = tf.compat.v2.where(tf.abs(y) > 1e-8)
    x = tf.gather_nd(y, indices)
    x = tf.cast(x, tf.int32)  # Empty tensor.
    assert x.shape.as_list() == [None]
    stage = self.default_encoding_stage()
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)

    test_data = self.evaluate_test_data(
        test_utils.TestData(x, encoded_x, decoded_x))
    assert test_data.x.shape == (0,)
    assert test_data.encoded_x[stage.ENCODED_VALUES_KEY].shape == (0,)
    assert test_data.decoded_x.shape == (0,)

  @parameterized.parameters([tf.bool, tf.float32])
  def test_encode_unsupported_type_raises(self, dtype):
    stage = self.default_encoding_stage()
    with self.assertRaisesRegexp(TypeError, 'Unsupported input type'):
      self.run_one_to_many_encode_decode(
          stage, lambda: tf.cast(self.default_input(), dtype))

  def test_encode_unsupported_input_shape_raises(self):
    x = tf.random.uniform((3, 4), maxval=10, dtype=tf.int32)
    stage = self.default_encoding_stage()
    params, _ = stage.get_params()
    with self.assertRaisesRegexp(ValueError, 'Number of dimensions must be 1'):
      stage.encode(x, params)


if __name__ == '__main__':
  tf.test.main()
