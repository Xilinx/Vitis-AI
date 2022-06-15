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

from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research import kashin
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing import test_utils


if tf.executing_eagerly():
  tf.compat.v1.disable_eager_execution()


class KashinHadamardEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return kashin.KashinHadamardEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.normal([3, 12])

  @property
  def is_lossless(self):
    """See base class."""
    return False

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertLen(
        data.encoded_x[
            kashin.KashinHadamardEncodingStage.ENCODED_VALUES_KEY].shape,
        2)
    # No other common asserts, as based on input properties, this transformation
    # can be lossy or lossless.

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
        kashin.KashinHadamardEncodingStage.ENCODED_VALUES_KEY]
    encoded_x_2 = test_data_2.encoded_x[
        kashin.KashinHadamardEncodingStage.ENCODED_VALUES_KEY]
    self.assertGreater(np.linalg.norm(encoded_x_1 - encoded_x_2), 0.1)

  @parameterized.parameters(
      [((4,), (1, 4)),
       ((6,), (1, 8)),
       ((1, 5), (1, 8)),
       ((1, 11), (1, 16)),
       ((1, 20), (1, 32)),
       ((1, 111), (1, 128)),
       ((2, 7), (2, 8)),
       ((4, 1), (4, 1)),
       ((9, 7), (9, 8))])
  def test_with_multiple_input_shapes_pad_1_0(self, input_dims,
                                              expected_output_dims):
    stage = kashin.KashinHadamardEncodingStage(pad_extra_level_threshold=1.0)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.random.normal(input_dims))
    self.common_asserts_for_test_data(test_data)
    # Make sure output shape is as expected.
    self.assertEqual(
        expected_output_dims, test_data.encoded_x[
            kashin.KashinHadamardEncodingStage.ENCODED_VALUES_KEY].shape)

  @parameterized.parameters(
      [((4,), (1, 8)),
       ((6,), (1, 8)),
       ((1, 5), (1, 8)),
       ((1, 11), (1, 16)),
       ((1, 20), (1, 32)),
       ((1, 111), (1, 256)),
       ((2, 7), (2, 16)),
       ((4, 1), (4, 2)),
       ((9, 7), (9, 16))])
  def test_with_multiple_input_shapes_pad_0_8(self, input_dims,
                                              expected_output_dims):
    stage = kashin.KashinHadamardEncodingStage(pad_extra_level_threshold=0.8)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.random.normal(input_dims))
    self.common_asserts_for_test_data(test_data)
    # Make sure output shape is as expected.
    self.assertEqual(
        expected_output_dims, test_data.encoded_x[
            kashin.KashinHadamardEncodingStage.ENCODED_VALUES_KEY].shape)

  @parameterized.parameters([True, False])
  def test_all_zero_input_works(self, last_iter_clip):
    # Tests that encoding does not blow up with all-zero input.
    stage = kashin.KashinHadamardEncodingStage(
        num_iters=3, eta=0.9, delta=1.0, last_iter_clip=last_iter_clip)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.zeros([3, 12]))
    self.common_asserts_for_test_data(test_data)
    self.assertAllEqual(
        np.zeros((3, 12)).astype(np.float32), test_data.decoded_x)

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
        kashin.KashinHadamardEncodingStage.ENCODED_VALUES_KEY].shape
    self.assertEqual(test_data.x.shape[0], encoded_shape[0])
    self.assertEqual(8, encoded_shape[1])

  def test_larger_num_iters_improves_accuracy(self):
    # If last_iter_clip = True, this potentially computes lossy representation.
    # Set delta large, to measure the effect of changing num_iters on accuracy.
    x = np.random.randn(3, 12).astype(np.float32)
    errors = []
    seed = tf.constant([1, 2], tf.int64)
    for num_iters in [2, 3, 4, 5]:
      stage = kashin.KashinHadamardEncodingStage(
          num_iters=num_iters, eta=0.9, delta=100.0, last_iter_clip=True)
      encode_params, decode_params = stage.get_params()
      # To keep the experiment consistent, we always need to use fixed seed.
      encode_params[kashin.KashinHadamardEncodingStage.SEED_PARAMS_KEY] = seed
      decode_params[kashin.KashinHadamardEncodingStage.SEED_PARAMS_KEY] = seed
      encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                  decode_params)
      test_data = test_utils.TestData(x, encoded_x, decoded_x)
      test_data = self.evaluate_test_data(test_data)
      errors.append(np.linalg.norm(test_data.x - test_data.decoded_x))
    for e1, e2 in zip(errors[:-1], errors[1:]):
      # The incurred error with less iterations should be greater.
      self.assertGreater(e1, e2)

  def test_last_iter_clip_false_is_lossless(self):
    # Make sure to set delta to something large so that there is something to
    # clip in the last iteration. Otherwise the test does not make sense.
    stage = kashin.KashinHadamardEncodingStage(
        num_iters=2, eta=0.9, delta=100.0, last_iter_clip=False)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.random.normal([3, 12]))
    self.assertAllClose(test_data.x, test_data.decoded_x)

  def test_eta_delta_take_tf_values(self):
    x = self.default_input()
    stage = kashin.KashinHadamardEncodingStage(
        eta=tf.constant(0.9), delta=tf.constant(1.0))
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = test_utils.TestData(x, encoded_x, decoded_x)
    self.generic_asserts(test_data, stage)
    self.common_asserts_for_test_data(test_data)

  @parameterized.parameters(
      itertools.product([tf.float32, tf.float64],
                        [tf.float32, tf.float64],
                        [tf.float32, tf.float64]))
  def test_input_types(self, x_dtype, eta_dtype, delta_dtype):
    stage = kashin.KashinHadamardEncodingStage(
        eta=tf.constant(0.9, eta_dtype), delta=tf.constant(1.0, delta_dtype))
    x = tf.random.normal([3, 12], dtype=x_dtype)
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = test_utils.TestData(x, encoded_x, decoded_x)
    test_data = self.evaluate_test_data(test_data)
    self.assertAllEqual(test_data.x.shape, test_data.decoded_x.shape)

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

  @parameterized.parameters([0.0, 1.0, -1.0, 2.5])
  def test_eta_out_of_bounds_raises(self, eta):
    with self.assertRaisesRegexp(ValueError, 'between 0 and 1'):
      kashin.KashinHadamardEncodingStage(eta=eta)

  @parameterized.parameters([0.0, -1.0])
  def test_delta_small_raises(self, delta):
    with self.assertRaisesRegexp(ValueError, 'greater than 0'):
      kashin.KashinHadamardEncodingStage(delta=delta)

  @parameterized.parameters([0, -1, -10])
  def test_num_iters_small_raises(self, num_iters):
    with self.assertRaisesRegexp(ValueError, 'positive'):
      kashin.KashinHadamardEncodingStage(num_iters=num_iters)

  def test_num_iters_tensor_raises(self):
    with self.assertRaisesRegexp(ValueError, 'num_iters'):
      kashin.KashinHadamardEncodingStage(
          num_iters=tf.constant(2, dtype=tf.int32))

  def test_last_iter_clip_tensor_raises(self):
    with self.assertRaisesRegexp(ValueError, 'last_iter_clip'):
      kashin.KashinHadamardEncodingStage(
          last_iter_clip=tf.constant(True, dtype=tf.bool))

  @parameterized.parameters([0, 1, 0.0, 1.0])
  def test_last_iter_clip_not_bool_raises(self, last_iter_clip):
    with self.assertRaisesRegexp(ValueError, 'last_iter_clip must be a bool'):
      kashin.KashinHadamardEncodingStage(last_iter_clip=last_iter_clip)

  def test_pad_extra_level_threshold_tensor_raises(self):
    with self.assertRaisesRegexp(ValueError, 'pad_extra_level_threshold'):
      kashin.KashinHadamardEncodingStage(
          pad_extra_level_threshold=tf.constant(0.8, dtype=tf.float32))


if __name__ == '__main__':
  tf.test.main()
