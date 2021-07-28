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
"""Quantizers specific to vitis behavior."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable
from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantizer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_ops

Quantizer = quantizer.Quantizer


@register_keras_serializable(package='Vitis', name='LastValueMinMaxQuantizer')
class LastValueMinMaxQuantizer(Quantizer):
  """Quantize tensor based on range the last batch of values."""

  def __init__(self, bit_width, round_mode):
    """Construct a LastValueQuantizer.

    This is an experimental API not subject to backward compatibility.

    Args:
      bit_width: Number of bits for quantization
    """
    self.bit_width = bit_width
    self.round_mode = round_mode

  def build(self, tensor_shape, name, layer):
    min_var = layer.add_weight(
        name + '_min',
        initializer=keras.initializers.Constant(-6.0),
        trainable=False)

    max_var = layer.add_weight(
        name + '_max',
        initializer=keras.initializers.Constant(6.0),
        trainable=False)

    self.weights = {
        'min_var': min_var,
        'max_var': max_var,
    }
    return self.weights

  def get_quantize_info(self):
    """Get current values of the weights"""
    quantize_info = {}
    for name, var in self.weights.items():
      quantize_info[name] = var.numpy()
    return quantize_info

  def set_quantize_info(self, new_quantize_info):
    """Set values of the weights"""
    for name, var in self.weights.items():
      if name in new_quantize_info:
        keras.backend.set_value(var, new_quantize_info[name])

  def __call__(self, inputs, is_training, mode, weights, **kwargs):
    """Quantize tensor.

    Args:
      inputs: Input tensor to be quantized.
      training: Whether the graph is currently training.
      weights: Dictionary of weights the quantizer can use to quantize the
        tensor. This contains the weights created in the `build` function.
      **kwargs: Additional variables which may be passed to the quantizer.

    Returns:
      Quantized tensor.
    """
    return vitis_quantize_ops.LastValueMinMaxQuantize(
        inputs,
        weights['min_var'],
        weights['max_var'],
        bit_width=self.bit_width,
        round_mode=self.round_mode,
        mode=mode,
        is_training=is_training)

  def get_config(self):
    return {
        'bit_width': self.bit_width,
        'round_mode': self.round_mode,
    }

  def __eq__(self, other):
    if not isinstance(other, LastValueMinMaxQuantizer):
      return False

    return (self.bit_width == other.bit_width and
            self.round_mode == other.round_mode)

  def __ne__(self, other):
    return not self.__eq__(other)


@register_keras_serializable(package='Vitis', name='LastValueQuantPosQuantizer')
class LastValueQuantPosQuantizer(Quantizer):
  """Quantize tensor based on range the last batch of values."""

  def __init__(self, bit_width, method, round_mode):
    """Construct a LastValueQuantizer.

    This is an experimental API not subject to backward compatibility.

    Args:
      bit_width: Number of bits for quantization
      method: Quantize method, 0 for non_overflow and 1 for min_diffs
      round_mode; Round mode, 0 for std and 1 for dpu.
    """
    self.bit_width = bit_width
    self.method = method
    self.round_mode = round_mode

  def build(self, tensor_shape, name, layer):
    min_var = layer.add_weight(
        name + '_min',
        initializer=keras.initializers.Constant(-6.0),
        trainable=False)

    max_var = layer.add_weight(
        name + '_max',
        initializer=keras.initializers.Constant(6.0),
        trainable=False)

    quant_pos_var = layer.add_weight(
        name + '_pos',
        initializer=keras.initializers.Constant(0.0),
        trainable=False)

    self.weights = {
        'min_var': min_var,
        'max_var': max_var,
        'quant_pos_var': quant_pos_var
    }
    return self.weights

  def get_quantize_info(self):
    """Get current values of the weights"""
    quantize_info = {}
    for name, var in self.weights.items():
      quantize_info[name] = var.numpy()
    return quantize_info

  def set_quantize_info(self, new_quantize_info):
    """Set values of the weights"""
    for name, var in self.weights.items():
      if name in new_quantize_info:
        keras.backend.set_value(var, new_quantize_info[name])

  def __call__(self, inputs, is_training, mode, weights, **kwargs):
    """Quantize tensor.

    Args:
      inputs: Input tensor to be quantized.
      training: Whether the graph is currently training.
      weights: Dictionary of weights the quantizer can use to quantize the
        tensor. This contains the weights created in the `build` function.
      **kwargs: Additional variables which may be passed to the quantizer.

    Returns:
      Quantized tensor.
    """
    return vitis_quantize_ops.LastValueQuantPosQuantize(
        inputs,
        weights['quant_pos_var'],
        weights['min_var'],
        weights['max_var'],
        bit_width=self.bit_width,
        method=self.method,
        round_mode=self.round_mode,
        mode=mode,
        is_training=is_training)

  def get_config(self):
    return {
        'bit_width': self.bit_width,
        'method': self.method,
        'round_mode': self.round_mode,
    }

  def __eq__(self, other):
    if not isinstance(other, LastValueQuantPosQuantizer):
      return False

    return (self.bit_width == other.bit_width and
            self.method == other.method and self.round_mode == other.round_mode)

  def __ne__(self, other):
    return not self.__eq__(other)


@register_keras_serializable(package='Vitis', name='LastValueLogThQuantizer')
class LastValueLogThQuantizer(Quantizer):
  """Quantize tensor based on range the last batch of values."""

  def __init__(self, bit_width, method, round_mode):
    """Construct a LastValueQuantizer.

    This is an experimental API not subject to backward compatibility.

    Args:
      bit_width: Number of bits for quantization
      method: Quantize method, 0 for non_overflow and 1 for min_diffs
      round_mode; Round mode, 0 for std and 1 for dpu.
    """
    self.bit_width = bit_width
    self.method = method
    self.round_mode = round_mode

  def build(self, tensor_shape, name, layer):
    min_var = layer.add_weight(
        name + '_min',
        initializer=keras.initializers.Constant(-6.0),
        trainable=False)

    max_var = layer.add_weight(
        name + '_max',
        initializer=keras.initializers.Constant(6.0),
        trainable=False)

    log_th_var = layer.add_weight(
        name + '_log_th',
        initializer=keras.initializers.Constant(0.0),
        trainable=True)

    self.weights = {
        'min_var': min_var,
        'max_var': max_var,
        'log_th_var': log_th_var
    }
    return self.weights

  def get_quantize_info(self):
    """Get current values of the weights"""
    quantize_info = {}
    for name, var in self.weights.items():
      quantize_info[name] = var.numpy()
    return quantize_info

  def set_quantize_info(self, new_quantize_info):
    """Set values of the weights"""
    for name, var in self.weights.items():
      if name in new_quantize_info:
        keras.backend.set_value(var, new_quantize_info[name])

  def __call__(self, inputs, is_training, mode, weights, **kwargs):
    """Quantize tensor.

    Args:
      inputs: Input tensor to be quantized.
      training: Whether the graph is currently training.
      weights: Dictionary of weights the quantizer can use to quantize the
        tensor. This contains the weights created in the `build` function.
      **kwargs: Additional variables which may be passed to the quantizer.

    Returns:
      Quantized tensor.
    """
    return vitis_quantize_ops.LastValueLogThQuantize(
        inputs,
        weights['log_th_var'],
        weights['min_var'],
        weights['max_var'],
        bit_width=self.bit_width,
        method=self.method,
        round_mode=self.round_mode,
        mode=mode,
        is_training=is_training)

  def get_config(self):
    return {
        'bit_width': self.bit_width,
        'method': self.method,
        'round_mode': self.round_mode,
    }

  def __eq__(self, other):
    if not isinstance(other, LastValueLogThQuantizer):
      return False

    return (self.bit_width == other.bit_width and
            self.method == other.method and self.round_mode == other.round_mode)

  def __ne__(self, other):
    return not self.__eq__(other)


def _types_dict():
  return {
      'LastValueMinMaxQuantizer': LastValueMinMaxQuantizer,
      'LastValueQuantPosQuantizer': LastValueQuantPosQuantizer,
      'LastValueLogThQuantizer': LastValueLogThQuantizer,
  }
