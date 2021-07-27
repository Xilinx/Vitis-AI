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
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

__all__ = ['VitisAveragePooling2D', 'VitisGlobalAveragePooling2D']

serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
logger = common_utils.VAILogger


@tf.function
def _get_avgpool_scale(kw, kh):
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

  def build(self, input_shape):
    super(VitisGlobalAveragePooling2D, self).build(input_shape)

  def call(self, inputs):
    outputs = super(VitisGlobalAveragePooling2D, self).call(inputs)

    # Simulate DPU hahavior of AvgPooling
    input_shape = array_ops.shape(inputs)
    rescale_factor = _get_avgpool_scale(input_shape[1], input_shape[2])

    if rescale_factor != 1.0:
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

  def build(self, input_shape):
    super(VitisAveragePooling2D, self).build(input_shape)
    # Compute rescale factor in build() since the pool_size is determined.
    self.rescale_factor = _get_avgpool_scale(self.pool_size[0],
                                             self.pool_size[1])

  def call(self, inputs):
    outputs = super(VitisAveragePooling2D, self).call(inputs)

    # Simulate DPU hahavior of AvgPooling
    input_shape = array_ops.shape(inputs)

    if self.rescale_factor != 1.0:
      outputs *= self.rescale_factor
    return outputs


def _types_dict():
  return {
      'VitisAveragePooling2D': VitisAveragePooling2D,
      'VitisGlobalAveragePooling2D': VitisGlobalAveragePooling2D,
  }
