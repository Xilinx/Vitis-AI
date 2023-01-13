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

from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research import clipping
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing import test_utils


if tf.executing_eagerly():
  tf.compat.v1.disable_eager_execution()


class ClipByNormEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return clipping.ClipByNormEncodingStage(1.0)

  def default_input(self):
    """See base class."""
    return tf.random.normal([20])

  @property
  def is_lossless(self):
    """See base class."""
    return False

  def common_asserts_for_test_data(self, data):
    """See base class."""
    encoded_x = data.encoded_x[
        clipping.ClipByNormEncodingStage.ENCODED_VALUES_KEY]
    # The encoding should not change the shape...
    self.assertAllEqual(data.x.shape, encoded_x.shape)
    # The decoding should be identity.
    self.assertAllEqual(encoded_x, data.decoded_x)

  def test_clipping_effective(self):
    stage = clipping.ClipByNormEncodingStage(1.0)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.constant([1.0, 1.0, 1.0, 1.0]))
    self.common_asserts_for_test_data(test_data)
    self.assertAllEqual([1.0, 1.0, 1.0, 1.0], test_data.x)
    # The decoded values should have norm 1.
    self.assertAllClose([0.5, 0.5, 0.5, 0.5], test_data.decoded_x)

  def test_clipping_large_norm_identity(self):
    stage = clipping.ClipByNormEncodingStage(1000.0)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.constant([1.0, 1.0, 1.0, 1.0]))
    self.common_asserts_for_test_data(test_data)
    # The encoding should act as an identity, if input value has smaller norm.
    self.assertAllEqual(test_data.x, test_data.decoded_x)

  @parameterized.parameters(([2,],), ([2, 3],), ([2, 3, 4],))
  def test_different_shapes(self, shape):
    stage = clipping.ClipByNormEncodingStage(1.0)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.random.uniform(shape) + 1.0)
    self.common_asserts_for_test_data(test_data)
    self.assertAllClose(1.0, np.linalg.norm(test_data.decoded_x))

  @parameterized.parameters(
      itertools.product([tf.float32, tf.float64], [tf.float32, tf.float64]))
  def test_input_types(self, x_dtype, clip_norm_dtype):
    # Tests combinations of input dtypes.
    stage = clipping.ClipByNormEncodingStage(
        tf.constant(1.0, clip_norm_dtype))
    x = tf.constant([1.0, 1.0, 1.0, 1.0], dtype=x_dtype)
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = test_utils.TestData(x, encoded_x, decoded_x)
    test_data = self.evaluate_test_data(test_data)

    self.assertAllEqual([1.0, 1.0, 1.0, 1.0], test_data.x)
    # The decoded values should have norm 1.
    self.assertAllClose([0.5, 0.5, 0.5, 0.5], test_data.decoded_x)


class ClipByValueEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return clipping.ClipByValueEncodingStage(-1.0, 1.0)

  def default_input(self):
    """See base class."""
    return tf.random.normal([20])

  @property
  def is_lossless(self):
    """See base class."""
    return False

  def common_asserts_for_test_data(self, data):
    """See base class."""
    encoded_x = data.encoded_x[
        clipping.ClipByValueEncodingStage.ENCODED_VALUES_KEY]
    # The encoding should not change the shape...
    self.assertAllEqual(data.x.shape, encoded_x.shape)
    # The decoding should be identity.
    self.assertAllEqual(encoded_x, data.decoded_x)

  def test_clipping_effective(self):
    stage = clipping.ClipByValueEncodingStage(-1.0, 1.0)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0]))
    self.common_asserts_for_test_data(test_data)
    self.assertAllEqual([-2.0, -1.0, 0.0, 1.0, 2.0], test_data.x)
    self.assertAllClose([-1.0, -1.0, 0.0, 1.0, 1.0], test_data.decoded_x)

  def test_clipping_large_min_max_identity(self):
    stage = clipping.ClipByValueEncodingStage(-1000.0, 1000.0)
    test_data = self.run_one_to_many_encode_decode(stage, self.default_input)
    self.common_asserts_for_test_data(test_data)
    # The encoding should act as an identity, if input has smaller values.
    self.assertAllEqual(test_data.x, test_data.decoded_x)

  @parameterized.parameters(([2,],), ([2, 3],), ([2, 3, 4],))
  def test_different_shapes(self, shape):
    stage = clipping.ClipByValueEncodingStage(-1.0, 1.0)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.random.normal(shape))
    self.common_asserts_for_test_data(test_data)
    self.assertGreaterEqual(1.0, np.amax(test_data.decoded_x))
    self.assertLessEqual(-1.0, np.amin(test_data.decoded_x))

  @parameterized.parameters(
      itertools.product([tf.float32, tf.float64], [tf.float32, tf.float64],
                        [tf.float32, tf.float64]))
  def test_input_types(self, x_dtype, clip_value_min_dtype,
                       clip_value_max_dtype):
    # Tests combinations of input dtypes.
    stage = clipping.ClipByValueEncodingStage(
        tf.constant(-1.0, clip_value_min_dtype),
        tf.constant(1.0, clip_value_max_dtype))
    x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=x_dtype)
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = test_utils.TestData(x, encoded_x, decoded_x)
    test_data = self.evaluate_test_data(test_data)

    self.common_asserts_for_test_data(test_data)
    self.assertAllEqual([-2.0, -1.0, 0.0, 1.0, 2.0], test_data.x)
    self.assertAllClose([-1.0, -1.0, 0.0, 1.0, 1.0], test_data.decoded_x)


if __name__ == '__main__':
  tf.test.main()
