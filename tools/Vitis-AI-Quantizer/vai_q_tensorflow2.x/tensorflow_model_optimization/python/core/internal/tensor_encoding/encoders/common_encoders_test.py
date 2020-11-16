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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import core_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import gather_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import simple_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.encoders import common_encoders
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils import py_utils

_ENCODER_FNS = [
    common_encoders.identity,
    lambda: common_encoders.uniform_quantization(8),
    lambda: common_encoders.hadamard_quantization(8),
]


class EncoderLibraryTest(parameterized.TestCase):
  """Tests for the `common_encoders` methods."""

  @parameterized.parameters(_ENCODER_FNS)
  def test_as_simple_encoder(self, encoder_fn):
    encoder = common_encoders.as_simple_encoder(encoder_fn(),
                                                tf.TensorSpec((2,), tf.float32))
    self.assertIsInstance(encoder, simple_encoder.SimpleEncoder)

  @parameterized.parameters(None, [[]], 2.0, 'string')
  def test_as_simple_encoder_raises_encoder(self, not_an_encoder):
    with self.assertRaises(TypeError):
      common_encoders.as_simple_encoder(not_an_encoder,
                                        tf.TensorSpec((2,), tf.float32))

  @parameterized.parameters(None, [[]], 2.0, 'string')
  def test_as_simple_encoder_raises_tensorspec(self, not_a_tensorspec):
    with self.assertRaises(TypeError):
      common_encoders.as_simple_encoder(common_encoders.identity(),
                                        not_a_tensorspec)

  @parameterized.parameters(_ENCODER_FNS)
  def test_as_gather_encoder(self, encoder_fn):
    encoder = common_encoders.as_gather_encoder(encoder_fn(),
                                                tf.TensorSpec((2,), tf.float32))
    self.assertIsInstance(encoder, gather_encoder.GatherEncoder)

  @parameterized.parameters(None, [[]], 2.0, 'string')
  def test_as_gather_encoder_raises_encoder(self, not_an_encoder):
    with self.assertRaises(TypeError):
      common_encoders.as_gather_encoder(not_an_encoder,
                                        tf.TensorSpec((2,), tf.float32))

  @parameterized.parameters(None, [[]], 2.0, 'string')
  def test_as_gather_encoder_raises_tensorspec(self, not_a_tensorspec):
    with self.assertRaises(TypeError):
      common_encoders.as_gather_encoder(common_encoders.identity(),
                                        not_a_tensorspec)

  def test_identity(self):
    encoder = common_encoders.identity()
    self.assertIsInstance(encoder, core_encoder.Encoder)

    params, _ = encoder.get_params(encoder.initial_state())
    encoded_x, _, _ = encoder.encode(tf.constant(1.0), params)
    keys = [k for k, _ in py_utils.flatten_with_joined_string_paths(encoded_x)]
    self.assertSameElements(['identity_values'], keys)

  def test_uniform_quantization(self):
    encoder = common_encoders.uniform_quantization(8)
    self.assertIsInstance(encoder, core_encoder.Encoder)

    params, _ = encoder.get_params(encoder.initial_state())
    encoded_x, _, _ = encoder.encode(tf.constant(1.0), params)
    keys = [k for k, _ in py_utils.flatten_with_joined_string_paths(encoded_x)]
    self.assertSameElements([
        'flattened_values/min_max',
        'flattened_values/quantized_values/bitpacked_values'
    ], keys)

  def test_hadamard_quantization(self):
    encoder = common_encoders.hadamard_quantization(8)
    self.assertIsInstance(encoder, core_encoder.Encoder)

    params, _ = encoder.get_params(encoder.initial_state())
    encoded_x, _, _ = encoder.encode(tf.constant(1.0), params)
    keys = [k for k, _ in py_utils.flatten_with_joined_string_paths(encoded_x)]
    self.assertSameElements([
        'flattened_values/hadamard_values/min_max',
        'flattened_values/hadamard_values/quantized_values/bitpacked_values'
    ], keys)


if __name__ == '__main__':
  tf.test.main()
