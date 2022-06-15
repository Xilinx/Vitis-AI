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

from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages import stages_impl
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing import test_utils


if tf.executing_eagerly():
  tf.compat.v1.disable_eager_execution()


class IdentityEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return stages_impl.IdentityEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(data.x, data.decoded_x)
    self.assertAllClose(
        data.x,
        data.encoded_x[stages_impl.IdentityEncodingStage.ENCODED_VALUES_KEY])


class FlattenEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return stages_impl.FlattenEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([2, 3])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(data.x, data.decoded_x)
    self.assertLen(  # The Tensor rank of encoded_x should be 1.
        data.encoded_x[
            stages_impl.FlattenEncodingStage.ENCODED_VALUES_KEY].shape, 1)
    self.assertAllClose(
        data.x.flatten(),
        data.encoded_x[stages_impl.FlattenEncodingStage.ENCODED_VALUES_KEY])

  def test_one_to_many_with_unknown_shape(self):
    """Tests that encoding works with statically not known input shape."""

    def random_shape_2d_tensor():
      # Returns Tensor of shape [3, <unknown>]
      random_shape_vector = test_utils.get_tensor_with_random_shape()
      return tf.stack([random_shape_vector] * 3)

    test_data = self.run_one_to_many_encode_decode(
        self.default_encoding_stage(), random_shape_2d_tensor)
    self.common_asserts_for_test_data(test_data)


class HadamardEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return stages_impl.HadamardEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.normal([1, 12])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    encoded_x = data.encoded_x[
        stages_impl.HadamardEncodingStage.ENCODED_VALUES_KEY]
    self.assertAllClose(data.x, data.decoded_x)
    self.assertLen(encoded_x.shape, 2)
    # This is a rotation, hence, the norms should be the same.
    # If the input has dimension 1, the transform is applied to the whole input.
    # If the input has dimension 2, the transform is applied to every single
    # vector separately.
    if len(data.x.shape) == 1:
      self.assertAllClose(np.linalg.norm(data.x), np.linalg.norm(encoded_x))
    else:
      for x, y in zip(data.x, encoded_x):
        self.assertAllClose(np.linalg.norm(x), np.linalg.norm(y))

  def test_encoding_randomized(self):
    # The encoding stage declares a source of randomness (a random seed) in the
    # get_params method, and different encoding should be produced for each
    # random seed. This tests that this is the case, and that the encoding is
    # still lossless.
    stage = self.default_encoding_stage()
    x = np.random.randn(20).astype(np.float32)
    test_data_1 = self.run_one_to_many_encode_decode(stage, lambda: x)
    test_data_2 = self.run_one_to_many_encode_decode(stage, lambda: x)
    # Make sure we encode the same object.
    self.assertAllClose(test_data_1.x, test_data_2.x)
    self.assertAllClose(test_data_1.x, test_data_1.decoded_x)
    self.assertAllClose(test_data_2.x, test_data_2.decoded_x)
    encoded_x_1 = test_data_1.encoded_x[
        stages_impl.HadamardEncodingStage.ENCODED_VALUES_KEY]
    encoded_x_2 = test_data_2.encoded_x[
        stages_impl.HadamardEncodingStage.ENCODED_VALUES_KEY]
    self.assertGreater(np.linalg.norm(encoded_x_1 - encoded_x_2), 0.1)

  @parameterized.parameters(
      [((4,), (1, 4)),
       ((7,), (1, 8)),
       ((1, 5), (1, 8)),
       ((1, 11), (1, 16)),
       ((1, 20), (1, 32)),
       ((1, 111), (1, 128)),
       ((2, 7), (2, 8)),
       ((4, 1), (4, 1)),
       ((9, 7), (9, 8))])
  def test_with_multiple_input_shapes(self, input_dims, expected_output_dims):
    test_data = self.run_one_to_many_encode_decode(
        self.default_encoding_stage(), lambda: tf.random.normal(input_dims))
    self.common_asserts_for_test_data(test_data)
    # Make sure output shape is as expected.
    self.assertEqual(
        expected_output_dims, test_data.encoded_x[
            stages_impl.HadamardEncodingStage.ENCODED_VALUES_KEY].shape)

  def test_input_with_unknown_leading_dimension(self):

    def get_random_shape_input():
      # Returns a Tensor of shape (?, 6)
      return tf.map_fn(lambda x: x * tf.random.normal([6]),
                       test_utils.get_tensor_with_random_shape())

    # Validate the premise of the test.
    assert get_random_shape_input().shape.as_list() == [None, 6]
    test_data = self.run_one_to_many_encode_decode(
        self.default_encoding_stage(), get_random_shape_input)
    self.common_asserts_for_test_data(test_data)
    encoded_shape = test_data.encoded_x[
        stages_impl.HadamardEncodingStage.ENCODED_VALUES_KEY].shape
    self.assertEqual(test_data.x.shape[0], encoded_shape[0])
    self.assertEqual(8, encoded_shape[1])

  @parameterized.parameters([tf.float32, tf.float64])
  def test_input_types(self, x_dtype):
    test_data = self.run_one_to_many_encode_decode(
        self.default_encoding_stage(),
        lambda: tf.random.normal([1, 12], dtype=x_dtype))
    self.common_asserts_for_test_data(test_data)

  def test_unknown_shape_raises(self):
    x = test_utils.get_tensor_with_random_shape()
    stage = self.default_encoding_stage()
    params, _ = stage.get_params()
    with self.assertRaisesRegexp(ValueError, 'fully known'):
      stage.encode(x, params)

  @parameterized.parameters([((1, 1, 5),), ((1, 1, 1, 5),)])
  def test_more_than_two_ndims_raises(self, dims):
    x = tf.random.normal(dims)
    stage = self.default_encoding_stage()
    params, _ = stage.get_params()
    with self.assertRaisesRegexp(ValueError, 'must be 1 or 2.'):
      stage.encode(x, params)


class UniformQuantizationEncodingStageStageTest(
    test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return stages_impl.UniformQuantizationEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([50], minval=-1.0, maxval=1.0)

  @property
  def is_lossless(self):
    """See base class."""
    return False

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self._assert_is_integer_float(data.encoded_x[
        stages_impl.UniformQuantizationEncodingStage.ENCODED_VALUES_KEY])

  def _assert_is_integer_float(self, quantized_vals):
    """Asserts that float type values are integers."""
    assert quantized_vals.dtype == np.float32
    self.assertAllClose(quantized_vals,
                        tf.cast(tf.cast(quantized_vals, np.int32), np.float32))

  @parameterized.parameters(
      itertools.product([1, 2, 3, 4, 7, 8, 9, 16], [None, [-0.5, 0.5]]))
  def test_quantization_bits_stochastic_rounding(self, bits, min_max):
    stage = stages_impl.UniformQuantizationEncodingStage(
        bits=bits, min_max=min_max, stochastic=True)
    test_data = self.run_one_to_many_encode_decode(stage, self.default_input)
    self._assert_is_integer_float(test_data.encoded_x[
        stages_impl.UniformQuantizationEncodingStage.ENCODED_VALUES_KEY])
    # For stochastic rounding, the potential error incurred by quantization
    # is bounded by the range of the input values divided by the number of
    # quantization buckets.
    if min_max is None:
      self.assertAllClose(
          test_data.x, test_data.decoded_x, rtol=0.0, atol=2 / (2**bits - 1))
    else:
      self.assertAllClose(
          np.clip(test_data.x, -0.5, 0.5),
          test_data.decoded_x,
          rtol=0.0,
          atol=1 / (2**bits - 1))

  @parameterized.parameters(
      itertools.product([1, 2, 3, 4, 7, 8, 9, 16], [None, [-0.5, 0.5]]))
  def test_quantization_bits_deterministic_rounding(self, bits, min_max):
    stage = stages_impl.UniformQuantizationEncodingStage(
        bits=bits, min_max=min_max, stochastic=False)
    test_data = self.run_one_to_many_encode_decode(stage, self.default_input)
    self._assert_is_integer_float(test_data.encoded_x[
        stages_impl.UniformQuantizationEncodingStage.ENCODED_VALUES_KEY])
    # For deterministic rounding, the potential error incurred by quantization
    # is bounded by half of the range of the input values divided by the number
    # of quantization buckets.
    if min_max is None:
      self.assertAllClose(
          test_data.x, test_data.decoded_x, rtol=0.0, atol=1 / (2**bits - 1))
    else:
      self.assertAllClose(
          np.clip(test_data.x, -0.5, 0.5),
          test_data.decoded_x,
          rtol=0.0,
          atol=0.5 / (2**bits - 1))

  def test_quantization_empirically_unbiased(self):
    # Tests that the quantization with stochastic=True "seems" to be unbiased.
    # Executing the encoding and decoding many times, the average error should
    # be a lot larger than the error of average decoded value.
    x = tf.constant(np.random.rand((50)).astype(np.float32))
    stage = stages_impl.UniformQuantizationEncodingStage(
        bits=2, stochastic=True)
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = test_utils.TestData(x, encoded_x, decoded_x)
    test_data_list = [self.evaluate_test_data(test_data) for _ in range(200)]

    norm_errors = []
    errors = []
    for data in test_data_list:
      norm_errors.append(np.linalg.norm(data.x - data.decoded_x))
      errors.append(data.x - data.decoded_x)
    mean_of_errors = np.mean(norm_errors)
    error_of_mean = np.linalg.norm(np.mean(errors, axis=0))
    self.assertGreater(mean_of_errors, error_of_mean * 10)

  @parameterized.parameters(
      itertools.product([tf.float32, tf.float64], [tf.float32, tf.float64]))
  def test_input_types(self, x_dtype, min_max_dtype):
    # Tests combinations of input dtypes.
    stage = stages_impl.UniformQuantizationEncodingStage(
        bits=8, min_max=tf.constant([-1.0, 1.0], min_max_dtype))
    x = tf.random.normal([50], dtype=x_dtype)
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = test_utils.TestData(x, encoded_x, decoded_x)
    test_data = self.evaluate_test_data(test_data)

    self.assertLess(np.amin(test_data.x), -1.0)
    self.assertGreater(np.amax(test_data.x), 1.0)
    self.assertAllGreaterEqual(test_data.decoded_x, -1.0)
    self.assertAllLessEqual(test_data.decoded_x, 1.0)

  def test_all_zero_input_works(self):
    # Tests that encoding does not blow up with all-zero input. With
    # min_max=None, the derived min and max are identical, thus potential for
    # division by zero.
    stage = stages_impl.UniformQuantizationEncodingStage(bits=8, min_max=None)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.zeros([50]))
    self.assertAllEqual(np.zeros((50)).astype(np.float32), test_data.decoded_x)

  def test_commutes_with_sum_given_min_max(self):
    stage = stages_impl.UniformQuantizationEncodingStage(bits=8,
                                                         min_max=[-1.0, 1.0])
    input_values = self.evaluate([tf.random.normal([50]) for _ in range(3)])
    server_test_data, decode_params = self.run_many_to_one_encode_decode(
        stage, input_values)
    self.assert_commutes_with_sum(
        server_test_data,
        stage,
        decode_params,
        shape=input_values[0].shape)

  @parameterized.parameters([0, 17, -1, 1.5])
  def test_bits_out_of_range_raises(self, bits):
    with self.assertRaisesRegexp(ValueError, 'integer between 1 and 16'):
      stages_impl.UniformQuantizationEncodingStage(bits=bits)

  @parameterized.parameters([1.0, ([1.0, 2.0, 3.0],)])
  def test_bad_min_max_tensor_raises(self, bad_min_max):
    with self.assertRaisesRegexp(ValueError, r'shape \(2\)'):
      stages_impl.UniformQuantizationEncodingStage(
          min_max=tf.constant(bad_min_max))

  @parameterized.parameters([([1.0],), ([1.0, 2.0, 3.0],)])
  def test_bad_min_max_python_shape_raises(self, bad_min_max):
    with self.assertRaisesRegexp(ValueError, 'list with two elements'):
      stages_impl.UniformQuantizationEncodingStage(min_max=bad_min_max)

  @parameterized.parameters([([1.0, 1.0],), ([2.0, 1.0],)])
  def test_bad_min_max_python_values_raises(self, bad_min_max):
    with self.assertRaisesRegexp(ValueError, 'smaller than the second'):
      stages_impl.UniformQuantizationEncodingStage(min_max=bad_min_max)

  def test_stochastic_tensor_raises(self):
    with self.assertRaisesRegexp(TypeError, 'stochastic'):
      stages_impl.UniformQuantizationEncodingStage(
          stochastic=tf.constant(True, dtype=tf.bool))


class BitpackingEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return stages_impl.BitpackingEncodingStage(8)

  def default_input(self):
    """See base class."""
    return tf.cast(
        tf.random.uniform([50], minval=0, maxval=255, dtype=tf.int32),
        tf.float32)

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    encoded_x = data.encoded_x[
        stages_impl.BitpackingEncodingStage.ENCODED_VALUES_KEY]
    self.assertAllClose(data.x, data.decoded_x)
    self.assertEqual(np.int32, encoded_x.dtype)
    self.assertGreaterEqual(data.x.size, encoded_x.size)

  @parameterized.parameters(
      itertools.product([1, 2, 3, 4, 7, 8, 9, 16],
                        [(1,), (50,), (5, 5), (5, 6, 4)]))
  def test_is_lossless(self, bits, shape):
    # Tests that the encoding is lossless, for a variety of inputs.
    def x_fn():
      return tf.cast(
          tf.random.uniform(
              shape, minval=0, maxval=2**bits - 1, dtype=tf.int32), tf.float32)

    stage = stages_impl.BitpackingEncodingStage(bits)
    test_data = self.run_one_to_many_encode_decode(stage, x_fn)
    self.assertAllClose(test_data.x, test_data.decoded_x)
    self.assertEqual(test_data.x.dtype, test_data.decoded_x.dtype)

  @parameterized.parameters([
      (1, [[1 + 4 + 8]]),
      (2, [[1 + 4**2 + 4**3]]),
      (3, [[1 + 8**2 + 8**3]]),
      (4, [[1 + 16**2 + 16**3]]),
      (8, [[16842753], [0]])])
  def test_encoded_values_as_expected(self, bits, expected_bitpacked_values):
    # Tests that the packed values are as expected.
    stage = stages_impl.BitpackingEncodingStage(bits)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.constant([1.0, 0.0, 1.0, 1.0, 0.0], dtype=tf.float32))
    self.assertAllEqual(
        expected_bitpacked_values, test_data.encoded_x[
            stages_impl.BitpackingEncodingStage.ENCODED_VALUES_KEY])

  def test_float_types(self):
    # Tests that both float32 and float64 type work correctly.
    stage = self.default_encoding_stage()
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.cast(self.default_input(), tf.float32))
    self.assertAllClose(test_data.x, test_data.decoded_x)
    self.assertEqual(np.float32, test_data.decoded_x.dtype)

    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.cast(self.default_input(), tf.float64))
    self.assertAllClose(test_data.x, test_data.decoded_x)
    self.assertEqual(np.float64, test_data.decoded_x.dtype)

  def test_bad_input_executes(self):
    # Test that if input to encode is outside of the expected range, everything
    # still executes, but the result is not correct.
    x = np.array([2**9] * 5).astype(np.int32)
    stage = stages_impl.BitpackingEncodingStage(8)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.constant(x, tf.float32))
    self.assertNotAllClose(x, test_data.decoded_x.astype(np.int32))

  @parameterized.parameters([tf.bool, tf.int32])
  def test_encode_unsupported_type_raises(self, dtype):
    stage = self.default_encoding_stage()
    with self.assertRaisesRegexp(TypeError, 'Unsupported packing type'):
      self.run_one_to_many_encode_decode(
          stage, lambda: tf.cast(self.default_input(), dtype))

  def test_bad_input_bits_raises(self):
    with self.assertRaisesRegexp(TypeError, 'cannot be a TensorFlow value'):
      stages_impl.BitpackingEncodingStage(tf.constant(1, dtype=tf.int32))
    with self.assertRaisesRegexp(ValueError, 'between 1 and 16'):
      stages_impl.BitpackingEncodingStage(0)
    with self.assertRaisesRegexp(ValueError, 'between 1 and 16'):
      stages_impl.BitpackingEncodingStage(17)


if __name__ == '__main__':
  tf.test.main()
