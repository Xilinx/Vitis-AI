# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for QuantizeWrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras import quantizers

QuantizeLayer = quantize_layer.QuantizeLayer
deserialize_layer = tf.keras.layers.deserialize
serialize_layer = tf.keras.layers.serialize


class QuantizeLayerTest(tf.test.TestCase):

  def setUp(self):
    super(QuantizeLayerTest, self).setUp()
    self.quant_params = {
        'num_bits': 8,
        'narrow_range': False
    }
    self.quantizer = quantizers.LastValueQuantizer(
        per_axis=False, symmetric=True, **self.quant_params)

  def testQuantizesTensors(self):
    model = tf.keras.Sequential([
        QuantizeLayer(
            quantizer=self.quantizer,
            input_shape=(4,)
        )])

    x = np.random.rand(1, 4)
    quant_x = tf.quantization.fake_quant_with_min_max_vars(
        x, -6.0, 6.0, **self.quant_params)

    self.assertAllClose(self.evaluate(quant_x), model.predict(x))

  def testSerializationQuantizeLayer(self):
    layer = QuantizeLayer(
        quantizer=self.quantizer,
        input_shape=(4,))

    custom_objects = {
        'QuantizeLayer': QuantizeLayer,
        'LastValueQuantizer': quantizers.LastValueQuantizer
    }

    serialized_layer = serialize_layer(layer)
    with tf.keras.utils.custom_object_scope(custom_objects):
      layer_from_config = deserialize_layer(serialized_layer)

    self.assertEqual(layer_from_config.get_config(), layer.get_config())


if __name__ == '__main__':
  tf.test.main()
