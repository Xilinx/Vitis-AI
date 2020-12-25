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
"""Quantize Annotate Wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config as quantize_config_mod

keras = tf.keras
deserialize_layer = tf.keras.layers.deserialize
serialize_layer = tf.keras.layers.serialize


class QuantizeAnnotateTest(tf.test.TestCase):

  class TestQuantizeConfig(quantize_config_mod.QuantizeConfig):

    def get_weights_and_quantizers(self, layer):
      pass

    def get_activations_and_quantizers(self, layer):
      pass

    def set_quantize_weights(self, layer, quantize_weights):
      pass

    def set_quantize_activations(self, layer, quantize_activations):
      pass

    def get_output_quantizers(self, layer):
      pass

    def get_config(self):
      return {}

  def testAnnotatesKerasLayer(self):
    layer = keras.layers.Dense(5, activation='relu', input_shape=(10,))
    model = keras.Sequential([layer])

    quantize_config = self.TestQuantizeConfig()
    annotated_model = keras.Sequential([
        quantize_annotate.QuantizeAnnotate(
            layer, quantize_config=quantize_config, input_shape=(10,))
    ])

    annotated_layer = annotated_model.layers[0]
    self.assertEqual(layer, annotated_layer.layer)
    self.assertEqual(quantize_config, annotated_layer.quantize_config)

    # Annotated model should not affect computation. Returns same results.
    x_test = np.random.rand(10, 10)
    self.assertAllEqual(model.predict(x_test), annotated_model.predict(x_test))

  def testSerializationQuantizeAnnotate(self):
    input_shape = (2,)
    layer = keras.layers.Dense(3)
    wrapper = quantize_annotate.QuantizeAnnotate(
        layer=layer,
        quantize_config=self.TestQuantizeConfig(),
        input_shape=input_shape)

    custom_objects = {
        'QuantizeAnnotate': quantize_annotate.QuantizeAnnotate,
        'TestQuantizeConfig': self.TestQuantizeConfig
    }

    serialized_wrapper = serialize_layer(wrapper)
    with tf.keras.utils.custom_object_scope(custom_objects):
      wrapper_from_config = deserialize_layer(serialized_wrapper)

    self.assertEqual(wrapper_from_config.get_config(), wrapper.get_config())

  def testQuantizeAnnotate_FailsWithModel(self):
    layer = keras.layers.Dense(5, activation='relu', input_shape=(10,))
    model = keras.Sequential([layer])

    with self.assertRaises(ValueError):
      quantize_annotate.QuantizeAnnotate(model)

if __name__ == '__main__':
  tf.test.main()
