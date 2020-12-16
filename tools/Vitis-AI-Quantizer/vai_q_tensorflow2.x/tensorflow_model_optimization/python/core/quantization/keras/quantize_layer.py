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
"""Keras Layer which quantizes tensors.

Module: tfmot.quantization.keras
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.utils import tf_utils
from tensorflow_model_optimization.python.core.quantization.keras import quantizers

serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object


class QuantizeLayer(tf.keras.layers.Layer):
  """Emulate quantization of tensors passed through the layer."""

  def __init__(self, quantizer, **kwargs):
    """Create a QuantizeLayer.

    Args:
      quantizer: `Quantizer` used to quantize tensors.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    super(QuantizeLayer, self).__init__(**kwargs)

    if quantizer is None or not isinstance(quantizer, quantizers.Quantizer):
      raise ValueError('quantizer should not be None, and should be an instance'
                       'of `tfmot.quantization.keras.quantizers.Quantizer`.')

    self.quantizer = quantizer

  def build(self, input_shape):
    self.quantizer_vars = self.quantizer.build(
        input_shape, self.name, self)

    self.optimizer_step = self.add_weight(
        'optimizer_step',
        initializer=tf.keras.initializers.Constant(-1),
        dtype=tf.dtypes.int32,
        trainable=False)

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()

    def _make_quantizer_fn(train_var):
      def quantizer_fn():
        return self.quantizer(
            inputs, train_var,
            weights=self.quantizer_vars)

      return quantizer_fn

    return tf_utils.smart_cond(
        training, _make_quantizer_fn(True), _make_quantizer_fn(False))

  def get_config(self):
    base_config = super(QuantizeLayer, self).get_config()
    config = {
        'quantizer': serialize_keras_object(self.quantizer)
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    # Deserialization code should ensure Quantizer is in keras scope.
    quantizer = deserialize_keras_object(
        config.pop('quantizer'),
        module_objects=globals(),
        custom_objects=None)

    return cls(quantizer=quantizer, **config)
