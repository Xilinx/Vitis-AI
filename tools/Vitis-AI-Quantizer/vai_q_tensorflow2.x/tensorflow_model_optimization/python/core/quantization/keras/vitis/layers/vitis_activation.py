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

__all__ = ['VitisSigmoid']

serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
logger = common_utils.VAILogger


@register_keras_serializable(package='Vitis', name='VitisSigmoid')
class VitisSigmoid(tf.keras.layers.Layer):
  """Vitis sigmoid layer.

  This is an simplified sigmoid layer to mimic the hardware sigmoid layer behaviour.
  """

  def __init__(self, **kwargs):
    """Create a Vitis sigmoid Layer.

    Args:
      quantizer: `Quantizer` used to quantize tensors.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    super(VitisSigmoid, self).__init__(**kwargs)

  def call(self, inputs):

    def hard_sigmoid_dpu(x):
      """A hardware friendly version of sigmoid function.

         hard_sigmoid: out = relu6(x + 3.) * 1. / 6.
         hard_sigmoid_dpu: out = relu6(x + 3.) * 2731 / 2 ^ 14
      """
      x = tf.cast(x, tf.float32)
      x_out = tf.keras.activations.relu(x + 3, max_value=6.)
      x_out = x_out * 2731. / 16384.
      return x_out

    return hard_sigmoid_dpu(inputs)

  def get_config(self):
    return super(VitisSigmoid, self).get_config()

  @classmethod
  def from_config(cls, config):
    config = config.copy()
    return cls(**config)


def _types_dict():
  return {
      'VitisSigmoid': VitisSigmoid,
  }
