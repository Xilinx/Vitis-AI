# Copyright 2019 Xilinx Inc.
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
"""Vitis quantize layers."""

import tensorflow as tf

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable
from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantizer as quantizer_mod
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

__all__ = ['VitisQuantize']

serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
logger = common_utils.VAILogger


@register_keras_serializable(package='Vitis', name='VitisQuantize')
class VitisQuantize(tf.keras.layers.Layer):
  """Emulate quantization of tensors passed through the layer."""

  def __init__(self, quantizer, mode, **kwargs):
    """Create a VitisQuantize Layer.

    Args:
      quantizer: `Quantizer` used to quantize tensors.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    super(VitisQuantize, self).__init__(**kwargs)

    if quantizer is None or not isinstance(quantizer, quantizer_mod.Quantizer):
      logger.error(
          'quantizer should not be None, and should be an instance'
          'of `tfmot.quantization.keras.vitis.base.quantizer.Quantizer`.')

    self.quantizer = quantizer
    self._mode = mode

  def build(self, input_shape):
    self.quantizer_vars = self.quantizer.build(input_shape, self.name, self)

    self.optimizer_step = self.add_weight(
        'optimizer_step',
        initializer=tf.keras.initializers.Constant(-1),
        dtype=tf.dtypes.int32,
        trainable=False)

  def get_quantize_info(self):
    return {'type': 'input', 'info': self.quantizer.get_quantize_info()}

  def set_quantize_info(self, new_quantize_info):
    self.quantizer.set_quantize_info(new_quantize_info['info'])

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()

    def _make_quantizer_fn(train_var):

      def quantizer_fn():
        return self.quantizer(
            inputs, train_var, self.mode, weights=self.quantizer_vars)

      return quantizer_fn

    return tf_utils.smart_cond(training, _make_quantizer_fn(True),
                               _make_quantizer_fn(False))

  def get_config(self):
    base_config = super(VitisQuantize, self).get_config()
    config = {
        'quantizer': serialize_keras_object(self.quantizer),
        'mode': self.mode
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    # Deserialization code should ensure Quantizer is in keras scope.
    quantizer = deserialize_keras_object(
        config.pop('quantizer'), module_objects=globals(), custom_objects=None)

    mode = config.pop('mode')

    return cls(quantizer=quantizer, mode=mode, **config)

  @property
  def mode(self):
    return self._mode

  @mode.setter
  def mode(self, value):
    self._mode = value


# Deprecated, use VitisQuantize Layer instead
class Quantize(tf.keras.layers.Layer):
  """Quantize layer."""

  def __init__(self, bit_width, quantize_pos, **kwargs):
    super(Quantize, self).__init__()
    self.bit_width = bit_width
    self.quantize_pos = quantize_pos

  def call(self, inputs):
    lower_bound = -tf.math.pow(2.0, self.bit_width - 1)
    upper_bound = tf.math.pow(2.0, self.bit_width - 1) - 1
    lower_bound = tf.cast(lower_bound, tf.float32, name="lower_bound")
    upper_bound = tf.cast(upper_bound, tf.float32, name="upper_bound")

    step = tf.math.pow(
        2.0, tf.cast(-self.quantize_pos, tf.float32), name="step")

    divided = tf.math.divide(inputs, step, name="divided")
    rounded = tf.math.round(divided, name="rounded")
    quantized = tf.clip_by_value(
        rounded, lower_bound, upper_bound, name="quantized")
    return quantized

  def get_config(self):
    config = super(Quantize, self).get_config()
    config.update({
        "bit_width": self.bit_width,
        "quantize_pos": self.quantize_pos
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


# Deprecated, use VitisQuantize Layer instead
class Dequantize(tf.keras.layers.Layer):
  """Dequantize layer."""

  def __init__(self, bit_width, quantize_pos, **kwargs):
    super(Dequantize, self).__init__()
    self.bit_width = bit_width
    self.quantize_pos = quantize_pos

  def call(self, inputs):
    step = tf.math.pow(
        2.0, tf.cast(-self.quantize_pos, tf.float32), name="step")
    dequantized = tf.math.multiply(inputs, step, name="dequantized")
    return dequantized

  def get_config(self):
    config = super(Dequantize, self).get_config()
    config.update({
        "bit_width": self.bit_width,
        "quantize_pos": self.quantize_pos
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


def _types_dict():
  return {
      'VitisQuantize': VitisQuantize,
      'QuantizeLayer': VitisQuantize,
  }
