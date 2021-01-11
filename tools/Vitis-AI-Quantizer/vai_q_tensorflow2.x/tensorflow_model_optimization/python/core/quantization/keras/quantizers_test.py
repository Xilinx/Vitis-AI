# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Quantizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.quantization.keras import quantizers

deserialize_keras_object = tf.keras.utils.deserialize_keras_object
serialize_keras_object = tf.keras.utils.serialize_keras_object


@keras_parameterized.run_all_keras_modes
@parameterized.parameters(
    quantizers.LastValueQuantizer,
    quantizers.MovingAverageQuantizer,
    quantizers.AllValuesQuantizer)
class QuantizersTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(QuantizersTest, self).setUp()
    self.quant_params = {
        'num_bits': 8,
        'per_axis': False,
        'symmetric': False,
        'narrow_range': False
    }

  def _test_quantizer(self, quantizer):
    inputs = tf.Variable(
        np.array([[-1.0, 0.5], [0.0, 1.0]]),
        name='inputs',
        dtype=tf.dtypes.float32)
    min_var = tf.Variable(0.0)
    max_var = tf.Variable(0.0)

    weights = {'min_var': min_var, 'max_var': max_var}
    quant_tensor = quantizer(inputs, training=True, weights=weights)

    compat.initialize_variables(self)
    results = self.evaluate(quant_tensor)
    min_max_values = self.evaluate([min_var, max_var])

    # TODO(pulkitb): Assert on expected values for testing.
    # Since the underlying code is already tested in quant_ops_test.py, this
    # just ensures the Quantizers code is wired properly.
    print('Result: ', results)
    print('min_var: ', min_max_values[0])
    print('max_var: ', min_max_values[1])

  def testQuantizer(self, quantizer_type):
    quantizer = quantizer_type(**self.quant_params)

    self._test_quantizer(quantizer)

  def testSerialization(self, quantizer_type):
    quantizer = quantizer_type(**self.quant_params)

    expected_config = {
        'class_name': quantizer_type.__name__,
        'config': {
            'num_bits': 8,
            'per_axis': False,
            'symmetric': False,
            'narrow_range': False
        }
    }
    serialized_quantizer = serialize_keras_object(quantizer)

    self.assertEqual(expected_config, serialized_quantizer)

    quantizer_from_config = deserialize_keras_object(
        serialized_quantizer,
        module_objects=globals(),
        custom_objects=quantizers._types_dict())

    self.assertEqual(quantizer, quantizer_from_config)


if __name__ == '__main__':
  tf.test.main()
