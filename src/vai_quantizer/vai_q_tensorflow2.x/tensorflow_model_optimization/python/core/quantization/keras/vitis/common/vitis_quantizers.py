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
from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantizer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_ops
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common.entropy_percentile import calibrator_numpy

Quantizer = quantizer.Quantizer
register_keras_serializable = tf.keras.utils.register_keras_serializable


@register_keras_serializable(package='Vitis', name='FSQuantizer')
class FSQuantizer(Quantizer):
  """Quantizer with float value scales."""

  def __init__(self,
               bit_width,
               method,
               round_mode,
               symmetry=True,
               per_channel=False,
               channel_axis=-1,
               use_framework_quant=True,
               unsigned=False,
               narrow_range=False,
               method_percentile=99.99):
    """Construct a FSQuantizer.

    This is an experimental API not subject to backward compatibility.

    Args:
      bit_width: Number of bits for quantization
      round_mode: The rounding function used for quantization
      symmetry: Whether to apply symmetry quantization
      per_channel: Whether to apply per_channel quantization. The last dimension is regarded as channel.
      channel_axis: The axis of the channel, used with per_channel enabled. The last dimension is 
        regarded as channel axis and other dimension will be reduces by default.
      use_framework_quant: Bool, whether to use the tensorflow fake_quantize operations. If not, the custom
        quantize kernel will be used.
      unsigned: Bool, whether to use unsigned integer for quantization.
      narrow_range: Bool, whether to use the narrow quantization range
        [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
    """
    self.bit_width = bit_width
    self.method = method
    self.round_mode = round_mode
    self.symmetry = symmetry
    self.per_channel = per_channel
    self.channel_axis = channel_axis
    self.use_framework_quant = use_framework_quant
    self.unsigned = unsigned
    self.narrow_range = narrow_range
    self.method_percentile = method_percentile
    #self.histogram = calibrator_numpy.HistogramCalibrator(bit_width, None, False)
    #self.calib_hist = tf.Variable(0)
    #self.calib_bin_edges = tf.Variable(0.0)

  def build(self, tensor_shape, name, layer):
    shape = None

    if self.per_channel:
      input_dim = len(tensor_shape)
      channel_axis = self.channel_axis
      if channel_axis < 0:
        channel_axis += input_dim

      shape = list(tensor_shape)
      for i in range(input_dim):
        if i != channel_axis:
          shape[i] = 1

    min_var = layer.add_weight(
        shape=shape,
        name=name + '_min',
        initializer=keras.initializers.Constant(-0.0),
        trainable=False)

    max_var = layer.add_weight(
        shape=shape,
        name=name + '_max',
        initializer=keras.initializers.Constant(0.0),
        trainable=False)

    #histogram
    calib_hist = tf.Variable([0],
                             name=name + '_calib_hist',
                             shape=[None],
                             dtype=tf.int32,
                             trainable=False)
    #float array for width
    layer._non_trainable_weights.append(calib_hist)
    calib_bin_edges = tf.Variable([0.],
                                  name=name + '_calib_bin_edges',
                                  shape=[None],
                                  dtype=tf.float32,
                                  trainable=False)
    layer._non_trainable_weights.append(calib_bin_edges)

    self.weights = {
        'min_var': min_var,
        'max_var': max_var,
        'calib_hist': calib_hist,
        'calib_bin_edges': calib_bin_edges,
    }
    return self.weights

  def get_quantize_info(self):
    """Get current values of the weights"""
    quantize_info = {}
    for name, var in self.weights.items():
      quantize_info[name] = keras.backend.get_value(var)
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
    quantize_res = vitis_quantize_ops.FSQuantize(
        inputs,
        weights['min_var'],
        weights['max_var'],
        weights['calib_hist'],
        weights['calib_bin_edges'],
        bit_width=self.bit_width,
        method=self.method,
        round_mode=self.round_mode,
        mode=mode,
        is_training=is_training,
        symmetry=self.symmetry,
        per_channel=self.per_channel,
        channel_axis=self.channel_axis,
        use_framework_quant=self.use_framework_quant,
        unsigned=self.unsigned,
        narrow_range=self.narrow_range)
    return quantize_res

  def get_config(self):
    return {
        'bit_width': self.bit_width,
        'method': self.method,
        'round_mode': self.round_mode,
        'symmetry': self.symmetry,
        'per_channel': self.per_channel,
        'channel_axis': self.channel_axis,
        'use_framework_quant': self.use_framework_quant,
        'unsigned': self.unsigned,
        'narrow_range': self.narrow_range,
        'method_percentile': self.method_percentile
    }

  def __eq__(self, other):
    if not isinstance(other, FSQuantizer):
      return False

    return (self.bit_width == other.bit_width and
            self.method == other.method and
            self.round_mode == other.round_mode and
            self.symmetry == other.symmetry and
            self.per_channel == other.per_channel and
            self.channel_axis == other.channel_axis and
            self.use_framework_quant == other.use_framework_quant and
            self.unsigned == other.unsigned and
            self.narrow_range == other.narrow_range)

  def __ne__(self, other):
    return not self.__eq__(other)


@register_keras_serializable(package='Vitis', name='MAFSQuantizer')
class MAFSQuantizer(Quantizer):
  """Quantizer with moving averagey float value scales."""

  def __init__(self,
               bit_width,
               round_mode,
               symmetry=True,
               per_channel=False,
               channel_axis=-1,
               ema_decay=0.999,
               use_framework_quant=True,
               unsigned=False,
               narrow_range=False):
    """Construct a MAFSQuantizer.

    This is an experimental API not subject to backward compatibility.

    Args:
      bit_width: Number of bits for quantization
      round_mode: The rounding function used for quantization
      symmetry: Whether to apply symmetry quantization
      per_channel: Whether to apply per_channel quantization. The last dimension is regarded as channel.
      channel_axis: The axis of the channel, used with per_channel enabled. The last dimension is 
        regarded as channel axis and other dimension will be reduces by default.
      ema_decay: EMA decay parameter.
      use_framework_quant: Bool, whether to use the tensorflow fake_quantize operations. If not, the custom
        quantize kernel will be used.
      unsigned: Bool, whether to use unsigned integer for quantization.
      narrow_range: Bool, whether to use the narrow quantization range
        [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
    """
    self.bit_width = bit_width
    self.round_mode = round_mode
    self.symmetry = symmetry
    self.per_channel = per_channel
    self.channel_axis = channel_axis
    self.ema_decay = ema_decay
    self.use_framework_quant = use_framework_quant
    self.unsigned = unsigned
    self.narrow_range = narrow_range

  def build(self, tensor_shape, name, layer):
    shape = None

    if self.per_channel:
      input_dim = len(tensor_shape)
      channel_axis = self.channel_axis
      if channel_axis < 0:
        channel_axis += input_dim

      shape = list(tensor_shape)
      for i in range(input_dim):
        if i != channel_axis:
          shape[i] = 1

    min_var = layer.add_weight(
        shape=shape,
        name=name + '_min',
        initializer=keras.initializers.Constant(-0.0),
        trainable=False)

    max_var = layer.add_weight(
        shape=shape,
        name=name + '_max',
        initializer=keras.initializers.Constant(0.0),
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
      quantize_info[name] = keras.backend.get_value(var)
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
    return vitis_quantize_ops.MAFSQuantize(
        inputs,
        weights['min_var'],
        weights['max_var'],
        bit_width=self.bit_width,
        round_mode=self.round_mode,
        mode=mode,
        is_training=is_training,
        symmetry=self.symmetry,
        per_channel=self.per_channel,
        channel_axis=self.channel_axis,
        ema_decay=self.ema_decay,
        use_framework_quant=self.use_framework_quant,
        unsigned=self.unsigned,
        narrow_range=self.narrow_range)

  def get_config(self):
    return {
        'bit_width': self.bit_width,
        'round_mode': self.round_mode,
        'symmetry': self.symmetry,
        'per_channel': self.per_channel,
        'channel_axis': self.channel_axis,
        'ema_decay': self.ema_decay,
        'use_framework_quant': self.use_framework_quant,
        'unsigned': self.unsigned,
        'narrow_range': self.narrow_range
    }

  def __eq__(self, other):
    if not isinstance(other, MAFSQuantizer):
      return False

    return (self.bit_width == other.bit_width and
            self.round_mode == other.round_mode and
            self.symmetry == other.symmetry and
            self.per_channel == other.per_channel and
            self.channel_axis == other.channel_axis and
            self.ema_decay == other.ema_decay and
            self.use_framework_quant == other.use_framework_quant and
            self.unsigned == other.unsigned and
            self.narrow_range == other.narrow_range)

  def __ne__(self, other):
    return not self.__eq__(other)


@register_keras_serializable(package='Vitis', name='Pof2SQuantizer')
class Pof2SQuantizer(Quantizer):
  """Quantizer with power-of-2 value scales."""

  def __init__(self,
               bit_width,
               method,
               round_mode,
               symmetry=True,
               per_channel=False,
               channel_axis=-1,
               unsigned=False,
               narrow_range=False):
    """Construct a Posf2SQuantizer.

    This is an experimental API not subject to backward compatibility.

    Args:
      bit_width: Number of bits for quantization
      method: Quantize method, 0 for non_overflow and 1 for min_diffs
      round_mode; Round mode, 0 for std and 1 for dpu.
      symmetry: Whether to apply symmetry quantization
      per_channel: Whether to apply per_channel quantization. The last dimension is regarded as channel.
      channel_axis: The axis of the channel, used with per_channel enabled. The last dimension is 
        regarded as channel axis and other dimension will be reduces by default.
      unsigned: Bool, whether to use unsigned integer for quantization.
      narrow_range: Bool, whether to use the narrow quantization range
        [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
    """
    self.bit_width = bit_width
    self.method = method
    self.round_mode = round_mode
    self.symmetry = symmetry
    self.per_channel = per_channel
    self.channel_axis = channel_axis
    self.unsigned = unsigned
    self.narrow_range = narrow_range

  def build(self, tensor_shape, name, layer):
    shape = None

    if self.per_channel:
      input_dim = len(tensor_shape)
      channel_axis = self.channel_axis
      if channel_axis < 0:
        channel_axis += input_dim

      shape = list(tensor_shape)
      for i in range(input_dim):
        if i != channel_axis:
          shape[i] = 1

    min_var = layer.add_weight(
        shape=shape,
        name=name + '_min',
        initializer=keras.initializers.Constant(-0.0),
        trainable=False)

    max_var = layer.add_weight(
        shape=shape,
        name=name + '_max',
        initializer=keras.initializers.Constant(0.0),
        trainable=False)

    quant_pos_var = layer.add_weight(
        shape=shape,
        name=name + '_pos',
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
      quantize_info[name] = keras.backend.get_value(var)
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
    return vitis_quantize_ops.Pof2SQuantize(
        inputs,
        weights['quant_pos_var'],
        weights['min_var'],
        weights['max_var'],
        bit_width=self.bit_width,
        method=self.method,
        round_mode=self.round_mode,
        mode=mode,
        is_training=is_training,
        symmetry=self.symmetry,
        per_channel=self.per_channel,
        channel_axis=self.channel_axis,
        unsigned=self.unsigned,
        narrow_range=self.narrow_range)

  def get_config(self):
    return {
        'bit_width': self.bit_width,
        'method': self.method,
        'round_mode': self.round_mode,
        'symmetry': self.symmetry,
        'per_channel': self.per_channel,
        'channel_axis': self.channel_axis,
        'unsigned': self.unsigned,
        'narrow_range': self.narrow_range,
    }

  def convert_to_fs_quantizer(self, use_framework_quant=True):
    config = self.get_config()
    config['use_framework_quant'] = use_framework_quant
    # Set round_mode to 1 as tf.fake only support HALF_UP rounding
    if use_framework_quant and config['round_mode'] != 1:
      config['round_mode'] = 1
    return FSQuantizer.from_config(config)

  def __eq__(self, other):
    if not isinstance(other, Pof2SQuantizer):
      return False

    return (self.bit_width == other.bit_width and
            self.method == other.method and
            self.round_mode == other.round_mode and
            self.symmetry == other.symmetry and
            self.per_channel == other.per_channel and
            self.channel_axis == other.channel_axis and
            self.unsigned == other.unsigned and
            self.narrow_range == other.narrow_range)

  def __ne__(self, other):
    return not self.__eq__(other)


@register_keras_serializable(package='Vitis', name='TQTQuantizer')
class TQTQuantizer(Quantizer):
  """Quantizer with power-of-2 value scales, using trained quantization thresholds."""

  def __init__(self,
               bit_width,
               method,
               round_mode,
               symmetry=True,
               per_channel=False,
               channel_axis=-1,
               unsigned=False,
               narrow_range=False):
    """Construct a TQTQuantizer.

    This is an experimental API not subject to backward compatibility.

    Args:
      bit_width: Number of bits for quantization
      method: Quantize method, 0 for non_overflow and 1 for min_diffs
      round_mode; Round mode, 0 for std and 1 for dpu.
      symmetry: Whether to apply symmetry quantization
      per_channel: Whether to apply per_channel quantization. The last dimension is regarded as channel.
      channel_axis: The axis of the channel, used with per_channel enabled. The last dimension is 
        regarded as channel axis and other dimension will be reduces by default.
      unsigned: Bool, whether to use unsigned integer for quantization.
      narrow_range: Bool, whether to use the narrow quantization range
        [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
    """
    self.bit_width = bit_width
    self.method = method
    self.round_mode = round_mode
    self.symmetry = symmetry
    self.per_channel = per_channel
    self.channel_axis = channel_axis
    self.unsigned = unsigned
    self.narrow_range = narrow_range

  def build(self, tensor_shape, name, layer):
    shape = None

    if self.per_channel:
      input_dim = len(tensor_shape)
      channel_axis = self.channel_axis
      if channel_axis < 0:
        channel_axis += input_dim

      shape = list(tensor_shape)
      for i in range(input_dim):
        if i != channel_axis:
          shape[i] = 1

    min_var = layer.add_weight(
        shape=shape,
        name=name + '_min',
        initializer=keras.initializers.Constant(-0.0),
        trainable=False)

    max_var = layer.add_weight(
        shape=shape,
        name=name + '_max',
        initializer=keras.initializers.Constant(0.0),
        trainable=False)

    log_th_var = layer.add_weight(
        shape=shape,
        name=name + '_log_th',
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
      quantize_info[name] = keras.backend.get_value(var)
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
    return vitis_quantize_ops.TQTQuantize(
        inputs,
        weights['log_th_var'],
        weights['min_var'],
        weights['max_var'],
        bit_width=self.bit_width,
        method=self.method,
        round_mode=self.round_mode,
        mode=mode,
        is_training=is_training,
        symmetry=self.symmetry,
        per_channel=self.per_channel,
        channel_axis=self.channel_axis,
        unsigned=self.unsigned,
        narrow_range=self.narrow_range)

  def get_config(self):
    return {
        'bit_width': self.bit_width,
        'method': self.method,
        'round_mode': self.round_mode,
        'symmetry': self.symmetry,
        'per_channel': self.per_channel,
        'channel_axis': self.channel_axis,
        'unsigned': self.unsigned,
        'narrow_range': self.narrow_range,
    }

  def convert_to_pof2s_quantizer(self):
    config = self.get_config()
    return Pof2SQuantizer.from_config(config)

  def __eq__(self, other):
    if not isinstance(other, TQTQuantizer):
      return False

    return (self.bit_width == other.bit_width and
            self.method == other.method and
            self.round_mode == other.round_mode and
            self.symmetry == other.symmetry and
            self.per_channel == other.per_channel and
            self.channel_axis == other.channel_axis and
            self.unsigned == other.unsigned and
            self.narrow_range == other.narrow_range)

  def __ne__(self, other):
    return not self.__eq__(other)


# Make aliases for backward compatibility
@register_keras_serializable(package='Vitis', name='LastValueMinMaxQuantizer')
class LastValueMinMaxQuantizer(FSQuantizer):
  pass


@register_keras_serializable(package='Vitis', name='MovingAvgMinMaxQuantizer')
class MovingAvgMinMaxQuantizer(MAFSQuantizer):
  pass


@register_keras_serializable(package='Vitis', name='LastValueQuantPosQuantizer')
class LastValueQuantPosQuantizer(Pof2SQuantizer):
  pass


@register_keras_serializable(package='Vitis', name='LastValueLogThQuantizer')
class LastValueLogThQuantizer(TQTQuantizer):
  pass


def _types_dict():
  return {
      'FSQuantizer': FSQuantizer,
      'MAFSQuantizer': MAFSQuantizer,
      'Pof2SQuantizer': Pof2SQuantizer,
      'TQTQuantizer': TQTQuantizer,
  }
