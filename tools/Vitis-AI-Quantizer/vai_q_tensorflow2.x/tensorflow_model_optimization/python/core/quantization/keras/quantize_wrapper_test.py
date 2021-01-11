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
"""Tests for QuantizeWrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry

QuantizeAwareActivation = quantize_aware_activation.QuantizeAwareActivation
QuantizeWrapper = quantize_wrapper.QuantizeWrapper
QuantizeRegistry = default_8bit_quantize_registry.QuantizeRegistry

keras = tf.keras
layers = tf.keras.layers

custom_object_scope = tf.keras.utils.custom_object_scope
deserialize_layer = tf.keras.layers.deserialize
serialize_layer = tf.keras.layers.serialize


class QuantizeWrapperTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(QuantizeWrapperTest, self).setUp()
    self.quantize_registry = QuantizeRegistry()

  def testQuantizesWeightsInLayer(self):
    weights = lambda shape, dtype: np.array([[-1.0, 0.0], [0.0, 1.0]])
    layer = keras.layers.Dense(2, kernel_initializer=weights)

    model = keras.Sequential([
        QuantizeWrapper(
            layer=layer,
            quantize_config=self.quantize_registry.get_quantize_config(layer),
            input_shape=(2,))
    ])

    # FakeQuant([-1.0, 1.0]) = [-0.9882355, 0.9882355]
    # Obtained from tf.fake_quant_with_min_max_vars
    self.assertAllClose(
        np.array([[-0.9882355, 0.9882355]]),
        # Inputs are all ones, so result comes directly from weights.
        model.predict(np.ones((1, 2))))

  # TODO(pulkitb): Extend test to support more layers.
  # The test validates several keras layers, but has limitations currently.
  #  1. Only layers with 'kernel' attribute work. Need to extend to others.
  #  2. Activations are not tested currently.
  #  3. RNN layers need to be added
  # TODO(tfmot): reenable some of these once added back to allowlist
  # after testing in quantize_functional_test.py.
  @parameterized.parameters(
      # (layers.Conv1D, (3, 6), {
      #     'filters': 4,
      #     'kernel_size': 2
      # }),
      (layers.Conv2D, (4, 6, 1), {
          'filters': 4,
          'kernel_size': (2, 2)
      }),
      # (layers.Conv2DTranspose, (7, 6, 3), {
      #     'filters': 2,
      #     'kernel_size': (3, 3)
      # }),
      # (layers.Conv3D, (5, 7, 6, 3), {
      #     'filters': 2,
      #     'kernel_size': (3, 3, 3)
      # }),
      # (layers.Conv3DTranspose, (5, 7, 6, 3), {
      #     'filters': 2,
      #    'kernel_size': (3, 3, 3)
      # }),
      # TODO(pulkitb): Add missing SeparableConv layers. The initializers are
      # different, so will need a change.
      (layers.Dense, (3,), {
          'units': 2
      }),
      # (layers.LocallyConnected1D, (3, 6), {
      #     'filters': 4,
      #     'kernel_size': 2
      # }),
      # (layers.LocallyConnected2D, (4, 6, 1), {
      #     'filters': 4,
      #     'kernel_size': (2, 2)
      # })
  )
  def testQuantizesWeights_KerasLayers(self, layer_type, input_shape, kwargs):
    self.weights = None

    def _get_random_weights(shape, dtype):  # pylint: disable=unused-argument
      self.weights = np.random.rand(*shape)
      return self.weights

    def _get_quantized_weights(shape, dtype):  # pylint: disable=unused-argument
      assert tuple(shape) == self.weights.shape

      # Default values used in DefaultRegistry.
      return tf.quantization.fake_quant_with_min_max_vars(
          self.weights, -6.0, 6.0, num_bits=8, narrow_range=True)

    layer = layer_type(kernel_initializer=_get_random_weights, **kwargs)
    quantized_model = keras.Sequential([
        QuantizeWrapper(
            layer=layer,
            quantize_config=self.quantize_registry.get_quantize_config(layer),
            input_shape=input_shape)
    ])
    # `model` gets constructed with same parameters as `quantized_model`. The
    # weights used are a quantized version of weights used in `quantized_model`.
    # This ensures the results of both the models should be the same since
    # quantization has been applied externally to `model`.
    model = keras.Sequential([
        layer_type(
            input_shape=input_shape,
            kernel_initializer=_get_quantized_weights,
            **kwargs)
    ])

    inputs = np.random.rand(1, *input_shape)
    # `quantized_model` should apply FakeQuant. Explicitly applying to the
    # results of `model` to verify QuantizeWrapper works as expected.
    expected_output = tf.quantization.fake_quant_with_min_max_vars(
        model.predict(inputs), -6.0, 6.0, num_bits=8, narrow_range=False)
    self.assertAllClose(expected_output, quantized_model.predict(inputs))

  def testQuantizesOutputsFromLayer(self):
    # TODO(pulkitb): Increase coverage by adding other output quantize layers
    # such as AveragePooling etc.

    layer = layers.ReLU()
    quantized_model = keras.Sequential([
        QuantizeWrapper(
            layers.ReLU(),
            quantize_config=self.quantize_registry.get_quantize_config(layer))
    ])

    model = keras.Sequential([layers.ReLU()])

    inputs = np.random.rand(1, 2, 1)
    expected_output = tf.quantization.fake_quant_with_min_max_vars(
        model.predict(inputs), -6.0, 6.0, num_bits=8, narrow_range=False)
    self.assertAllClose(expected_output, quantized_model.predict(inputs))

  def testSerializationQuantizeWrapper(self):
    input_shape = (2,)
    layer = keras.layers.Dense(3)
    wrapper = QuantizeWrapper(
        layer=layer,
        quantize_config=self.quantize_registry.get_quantize_config(layer),
        input_shape=input_shape)

    custom_objects = {
        'QuantizeAwareActivation': QuantizeAwareActivation,
        'QuantizeWrapper': QuantizeWrapper
    }
    custom_objects.update(default_8bit_quantize_registry._types_dict())

    serialized_wrapper = serialize_layer(wrapper)
    with custom_object_scope(custom_objects):
      wrapper_from_config = deserialize_layer(serialized_wrapper)

    self.assertEqual(wrapper_from_config.get_config(), wrapper.get_config())

  def testQuantizeWrapper_FailsWithModel(self):
    layer = keras.layers.Dense(5, activation='relu', input_shape=(10,))
    model = keras.Sequential([layer])

    with self.assertRaises(ValueError):
      QuantizeWrapper(
          model,
          quantize_config=self.quantize_registry.get_quantize_config(layer))

  # TODO(pulkitb): Add test to ensure weights are also preserved.


if __name__ == '__main__':
  tf.test.main()
