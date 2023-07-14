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
"""Vitis activation layers."""

import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

__all__ = ['VitisAveragePooling2D', 'VitisGlobalAveragePooling2D']

register_keras_serializable = tf.keras.utils.register_keras_serializable
serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
logger = common_utils.VAILogger


def _gcd_f(a, b):
  """Get the greatest common dividisor."""
  return a if b == 0 else _gcd_f(b, a % b)


def _lcm_f(a, b):
  """Get the least common multiple."""
  return int((a * b) / _gcd_f(a, b))


def _get_dpu_kernel_size(kh, kw):
  """For global_average_pooling, DPU will do padding to replace rectangle 
  kernel to squre kernel."""
  new_k = _lcm_f(kh, kw)
  return new_k, new_k


@tf.function
def _get_avgpool_scale(kh, kw):
  if kh > 255 or kw > 255:
    return 1.0
  elif kh == 3 and kw == 3:
    return 9.0 * 7.0 / 64.0
  elif kh == 5 and kw == 5:
    return 25.0 * 10.0 / 256.0
  elif kh == 6 and kw == 6:
    return 36.0 * 7.0 / 256.0
  elif kh == 7 and kw == 7:
    return 49.0 * 21.0 / 1024.0
  elif kh == 14 and kw == 14:
    return 196.0 * 21.0 / 4096.0
  else:
    rec = tf.cast(kw * kh, tf.float32)
    n_max = 7 + tf.math.ceil(tf.math.log(rec) / tf.math.log(2.))
    ns = tf.range(0., n_max)
    ns_pow = tf.pow(2., ns)
    ks = tf.round(ns_pow / rec)
    diffs = tf.math.abs(ks / ns_pow - 1 / rec)
    n = tf.argmin(diffs)
    k = ks[n]
    scale = k / tf.pow(2., tf.cast(n, tf.float32))
    scale *= rec
    return scale


@register_keras_serializable(
    package='Vitis', name='VitisGlobalAveragePooling2D')
class VitisGlobalAveragePooling2D(tf.keras.layers.GlobalAveragePooling2D):
  """Vitis version of GlobalAveragePooling2D layer.

  This is an Vitis version of average pooling to simulate DPU behaviour which to
  integer approximations for averaging of specific sizes.
  """

  def __init__(self, **kwargs):
    """Create a Vitis.GlobalAveragePooling2D Layer.

    Args:
      quantizer: `Quantizer` used to quantize tensors.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    super(VitisGlobalAveragePooling2D, self).__init__(**kwargs)
    self.rescale_factor = None

  def build(self, input_shape):
    super(VitisGlobalAveragePooling2D, self).build(input_shape)
    # Simulate DPU hahavior of AvgPooling for static input shape
    kh, kw = input_shape[1], input_shape[2]
    if None not in [kh, kw]:
      if kh == kw:
        self.rescale_factor = _get_avgpool_scale(kh, kw)
      else:
        # Try to convert rectangle kernel to square if this is a global_average_pooling
        new_kh, new_kw = _get_dpu_kernel_size(kh, kw)
        # Now DPU only supports square kernel size <= 8
        if new_kh <= 8:
          logger.debug(
              'Convert GlobalAveragePooling2D layer {}\'s kernel from {} to {} to simulate DPU behavior.'
              .format(self.name, (kh, kw), (new_kh, new_kw)))
          self.rescale_factor = _get_avgpool_scale(new_kh, new_kw)
        else:
          self.rescale_factor = _get_avgpool_scale(kh, kw)

    if self.rescale_factor is not None:
      logger.debug(
          'Rescale GlobalAveragePooling2D layer {} kernel size {} with factor {} to simulate DPU behavior.'
          .format(self.name, (kh, kw), self.rescale_factor))

  def call(self, inputs):
    outputs = super(VitisGlobalAveragePooling2D, self).call(inputs)

    # Simulate DPU hahavior of AvgPooling for dynamic input shape
    if self.rescale_factor is None:
      input_shape = array_ops.shape(inputs)
      kh, kw = input_shape[1], input_shape[2]
      #TODO(Xiao) support rectangle kernel conversion for dynamic input shape
      rescale_factor = _get_avgpool_scale(kw, kh)
      #  tf.print('GlobalAveragePooling2D: ', self.name, ' k:', (kh, kw),
      #           ' rescale_factor:', rescale_factor)
    else:
      rescale_factor = self.rescale_factor

    outputs *= rescale_factor
    return outputs


@register_keras_serializable(package='Vitis', name='AveragePooling2D')
class VitisAveragePooling2D(tf.keras.layers.AveragePooling2D):
  """Vitis version of AveragePooling2D layer.

  This is an Vitis version of average pooling to simulate DPU behaviour which uses
  integer approximations for averaging of specific sizes.
  """

  def __init__(self, **kwargs):
    """Create a Vitis.AveragePooling2D Layer.

    Args:
      quantizer: `Quantizer` used to quantize tensors.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    super(VitisAveragePooling2D, self).__init__(**kwargs)

  def _is_global_pooling(self, input_shape):
    """Check if this average_pooling can be converted to global_average_pooling."""
    output_shape = self.compute_output_shape(input_shape).as_list()
    return output_shape[1] == 1 and output_shape[2] == 1

  def build(self, input_shape):
    super(VitisAveragePooling2D, self).build(input_shape)

    # Compute rescale factor in build() since the pool_size is determined.
    self.rescale_factor = None
    kh, kw = self.pool_size[0], self.pool_size[1]

    if kh == kw:
      self.rescale_factor = _get_avgpool_scale(kh, kw)
    else:
      # Try to convert rectangle kernel to square if this is a global_average_pooling
      if self._is_global_pooling(input_shape):
        new_kh, new_kw = _get_dpu_kernel_size(kh, kw)
        # Now DPU only supports square kernel size <= 8
        if new_kh <= 8:
          logger.debug(
              'Convert AveragePooling2D layer {}\'s kernel from {} to {} to simulate DPU behavior.'
              .format(self.name, (kh, kw), (new_kh, new_kw)))
          self.rescale_factor = _get_avgpool_scale(new_kh, new_kw)
        else:
          rescale_factor = _get_avgpool_scale(kh, kw)
      else:
        rescale_factor = _get_avgpool_scale(kh, kw)

    if self.rescale_factor is not None:
      logger.debug(
          'Rescale GlobalAveragePooling2D layer {} kernel size {} with factor {} to simulate DPU behavior.'
          .format(self.name, (kh, kw), self.rescale_factor))

  def call(self, inputs):
    outputs = super(VitisAveragePooling2D, self).call(inputs)

    # Simulate DPU hahavior of AvgPooling
    if self.rescale_factor is not None:
      outputs *= self.rescale_factor
    return outputs


def _types_dict():
  return {
      'VitisAveragePooling2D': VitisAveragePooling2D,
      'VitisGlobalAveragePooling2D': VitisGlobalAveragePooling2D,
  }
