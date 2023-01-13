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
"""Python implementation for quantization operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import enum
import functools
import tensorflow as tf
import numpy as np

from tensorflow.python.training import moving_averages
from tensorflow_model_optimization.python.core.keras import compat as tf_compat
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_round
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common.entropy_percentile import calibrator_numpy

logger = common_utils.VAILogger


class QuantizeMethod(enum.Enum):
  """Enum class for quantize methods."""

  # NON_OVERFLOW method, ensure no value overflows.
  NON_OVERFLOW = 0

  # MIN_MSE method, minimize the MSE of float and quantized value.
  MIN_MSE = 1

  MIN_KL = 2

  PERCENTILE = 3


def sym_quantize(inputs, scale, q_min, q_max, round_mode):
  """Symmetry Quantize Kernel.  Q(x) = q_min + round[(x * scale]."""
  with tf.name_scope("SymQuantize"):
    rounded = vitis_round.round(inputs * scale, round_mode)
    quantized = tf.clip_by_value(rounded, q_min, q_max)
  return quantized


def asym_quantize(inputs, scale, shift, q_min, q_max, round_mode):
  """Asymmetry Quantize Kernel.  Q(x) = q_min + round[(x - shift) * scale]."""
  with tf.name_scope("AsymQuantize"):
    rounded = vitis_round.round((inputs - shift) * scale, round_mode)
    quantized = tf.clip_by_value(q_min + rounded, q_min, q_max)
  return quantized


def quantize(inputs, scale, shift, q_min, q_max, round_mode, symmetry):
  """Quantize Kernel."""
  if symmetry:
    return sym_quantize(inputs, scale, q_min, q_max, round_mode)
  else:
    return asym_quantize(inputs, scale, shift, q_min, q_max, round_mode)


def sym_dequantize(inputs, scale, q_min, q_max):
  """Dequantize Kernel.  DQ(x) =  x / scale."""
  with tf.name_scope("SymDequantize"):
    return inputs / scale


def asym_dequantize(inputs, scale, shift, q_min, q_max):
  """Dequantize Kernel.  DQ(x) =  (x - q_min) / scale + shift."""
  with tf.name_scope("AsymDequantize"):
    return (inputs - q_min) / scale + shift


def dequantize(inputs, scale, shift, q_min, q_max, round_mode, symmetry):
  """Dequantize Kernel."""
  if symmetry:
    return sym_dequantize(inputs, scale, q_min, q_max)
  else:
    return asym_dequantize(inputs, scale, shift, q_min, q_max)


def quantize_zero_point(scale, f_min, f_max, q_min, q_max):
  """Quantize the zero point."""
  with tf.name_scope("QuantizeZeroPoint"):
    f_zero_point = q_min - f_min * scale

    below_min = (f_zero_point < q_min)
    above_max = (f_zero_point > q_max)
    # Use std round for all zero points
    q_zero_point = vitis_round.round(f_zero_point,
                                     vitis_round.RoundMode.HALF_AWAY_FROM_ZERO)
    q_zero_point = tf.where(below_min, q_min, q_zero_point)
    q_zero_point = tf.where(above_max, q_max, q_zero_point)

    new_f_min = (q_min - q_zero_point) / scale
    new_f_max = (q_max - q_zero_point) / scale

    return q_zero_point, new_f_min, new_f_max


def get_scale(f_min, f_max, q_min, q_max):
  """Get quantize scaling factor."""
  return (q_max - q_min) / (f_max - f_min)


def get_min_max(inputs,
                bit_width,
                method=0,
                symmetry=True,
                per_channel=False,
                unsigned=False,
                narrow_range=False,
                reduce_dims=None,
                hist=None,
                bin_edges=None):
  """Get minimum and maximum value of inputs."""
  input_shape = inputs.get_shape()
  input_dims = len(input_shape)
  if per_channel:
    if input_dims >= 2:
      batch_min = tf.math.reduce_min(
          inputs, axis=reduce_dims, keepdims=True, name='batch_min')
      batch_max = tf.math.reduce_max(
          inputs, axis=reduce_dims, keepdims=True, name='batch_max')
    else:
      batch_min = inputs
      batch_max = inputs
  else:
    batch_min = tf.math.reduce_min(inputs, name='batch_min')
    batch_max = tf.math.reduce_max(inputs, name='batch_max')

  if symmetry:
    if unsigned:
      input_min = tf.math.reduce_min(inputs)  # may be NaN tensor
      input_min = tf.cond(
          tf.math.is_nan(input_min), lambda: tf.constant(0.), lambda: input_min)
      tf.Assert(
          tf.greater_equal(input_min, 0.), [
              "Input tensor values should be non-negative in symmetric unsigned quantization",
              inputs, "and the minimum is", input_min
          ])
      range_min = tf.math.minimum(batch_min, 0.0, name='range_min')
      range_max = tf.math.maximum(batch_max, 0.0, name='range_max')
    elif narrow_range:
      range_min = tf.minimum(batch_min, -batch_max)
      range_max = tf.maximum(batch_max, -batch_min)
    else:
      # Use full range of bit_width, the negative range is slightly larger than the positive range.
      min_max_ratio = -((1 << bit_width) - 2) / (1 << bit_width)
      range_min = tf.minimum(batch_min, batch_max / min_max_ratio)
      range_max = tf.maximum(batch_max, batch_min * min_max_ratio)
  else:
    range_min = tf.math.minimum(batch_min, 0.0, name='range_min')
    range_max = tf.math.maximum(batch_max, 0.0, name='range_max')
  return range_min, range_max


class FSFakeQuantize(object):
  """The fake quantization with float scale operation kernel."""

  def __init__(self,
               bit_width,
               round_mode,
               symmetry,
               per_channel,
               use_framework_quant=True,
               unsigned=False,
               narrow_range=False,
               reduce_dims=None):
    """Initialize the fake quantize with float scale operation kernel.

    Args:
      bit_width: Number of bits to use for quantization, must be between 2 and 8.
      round_mode: Int Enum value of the rounding mode. 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO.
      symmetry: Bool, whether to use symmetry quantization.
      per_channel: Bool, whether to use per-channel quantization.
      use_framework_quant: Bool, whether to use the tensorflow fake_quantize operations. If not, the custom
        quantize kernel will be used.
      unsigned: Bool, whether to use unsigned integer for quantization.
      narrow_range: Bool, whether to use the narrow quantization range
        [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
      reduce_dims: List(Int) containing the dimensions to reduce, only take effect when per_channel=True.
    """
    self.bit_width = bit_width
    self.unsigned = unsigned
    self.narrow_range = False if self.unsigned else narrow_range
    if self.unsigned:
      bound = float(2**self.bit_width)
      self.q_min, self.q_max = 0., bound - 1
    else:
      bound = float(2**(self.bit_width - 1))
      self.q_min, self.q_max = -bound, bound - 1
      if self.narrow_range:
        self.q_min += 1

    self.round_mode = vitis_round.RoundMode(round_mode)
    self.symmetry = symmetry
    self.per_channel = per_channel
    self.reduce_dims = reduce_dims
    self.use_framework_quant = use_framework_quant

    # tf.fake_quant_with_min_max_vars donot support unsigned input setting
    if self.use_framework_quant and self.unsigned:
      logger.debug(
          'Use framework_fs_fake_quant_v2 for quantization when unsigned enabled.'
      )

    # tf.fake_quant_with_min_max_vars only support HALF_UP rounding
    if self.use_framework_quant and self.round_mode != vitis_round.RoundMode.HALF_UP:
      logger.error(
          'Cannot support round mode ({}) with tf.fake_quant, please set round_mode=1'
          'or set use_framework_quant=False.'.format(self.round_mode))

    self.perm = None
    if self.per_channel and self.use_framework_quant:
      input_dims = len(self.reduce_dims) + 1
      channel_axis = convert_reduce_dims_to_channel_axis(
          input_dims, self.reduce_dims)
      if channel_axis != input_dims - 1:
        self.perm = self.reduce_dims[:]
        self.perm.append(channel_axis)

  def framework_fs_fake_quant(self, inputs, f_min, f_max):
    """Tensorflow framework version of float scale fake quantization operation, using HALF_UP round mode.

    Args:
      inputs: a tensor containing values to be quantized.
      f_min: the minimum input value.
      f_max: the maximum input value.
    Returns:
      a tensor containing quantized values.
    """

    if self.per_channel:
      # tf.quantization.fake_quant_with_min_max_vars_per_channel accept squeezed min_max input.
      f_min = tf.squeeze(f_min)
      f_max = tf.squeeze(f_max)

      # Do transpose when channel_axis is not the last dimension
      if self.perm is not None:
        inputs = tf.transpose(inputs, self.perm)
        inputs = tf.quantization.fake_quant_with_min_max_vars_per_channel(
            inputs,
            f_min,
            f_max,
            num_bits=self.bit_width,
            narrow_range=self.narrow_range)
        return tf.transpose(inputs, self.perm)
      else:
        return tf.quantization.fake_quant_with_min_max_vars_per_channel(
            inputs,
            f_min,
            f_max,
            num_bits=self.bit_width,
            narrow_range=self.narrow_range)
    else:
      return tf.quantization.fake_quant_with_min_max_vars(
          inputs,
          f_min,
          f_max,
          num_bits=self.bit_width,
          narrow_range=self.narrow_range)

  def framework_fs_fake_quant_v2(self, inputs, f_min, f_max):
    """Tensorflow framework version of float scale fake quantization operation,
    using tf.quantization.quantize_and_dequantize_v2 op.

    Args:
      inputs: a tensor containing values to be quantized.
      f_min: the minimum input value.
      f_max: the maximum input value.
    Returns:
      a tensor containing quantized values.
    """

    if self.per_channel:
      # tf.quantization.quantize_and_dequantize_v2 accept squeezed min_max input.
      f_min = tf.squeeze(f_min)
      f_max = tf.squeeze(f_max)
      return tf.quantization.quantize_and_dequantize_v2(
          inputs,
          f_min,
          f_max,
          signed_input=(self.unsigned == False),
          num_bits=self.bit_width,
          range_given=True,
          round_mode='HALF_UP',
          narrow_range=self.narrow_range,
          axis=self.channel_axis)
    else:
      return tf.quantization.quantize_and_dequantize_v2(
          inputs,
          f_min,
          f_max,
          signed_input=(self.unsigned == False),
          num_bits=self.bit_width,
          range_given=True,
          round_mode='HALF_UP',
          narrow_range=self.narrow_range)

  @tf.custom_gradient
  def custom_fs_fake_quant(self, inputs, f_min, f_max):
    """The custom float scale fake quantization operation kernel.

    Args:
      inputs: a tensor containing values to be quantized.
      f_min: the minimum input value.
      f_max: the maximum input value.
    Returns:
      a tensor containing quantized values.
    """
    scale = get_scale(f_min, f_max, self.q_min, self.q_max)
    if self.symmetry:
      shift = 0
    else:
      q_zero_point, new_f_min, new_f_max = quantize_zero_point(
          scale, f_min, f_max, self.q_min, self.q_max)
      shift = new_f_min

    quantized = quantize(inputs, scale, shift, self.q_min, self.q_max,
                         self.round_mode, self.symmetry)
    dequantized = dequantize(quantized, scale, shift, self.q_min, self.q_max,
                             self.round_mode, self.symmetry)

    def grad_fn(dy):
      """Custom gradient function."""
      if self.symmetry:
        _f_min, _f_max = f_min, f_max
      else:
        _f_min, _f_max = new_f_min, new_f_max

      between_min_max = (inputs >= _f_min) & (inputs <= _f_max)
      below_min = (inputs < _f_min)
      above_max = (inputs > _f_max)

      ones = tf.ones_like(dy)
      zeros = tf.zeros_like(dy)
      grad_wrt_inputs = dy * tf.where(between_min_max, ones, zeros)
      if self.per_channel:
        grad_wrt_f_min = tf.reduce_sum(
            dy * tf.where(below_min, ones, zeros),
            self.reduce_dims,
            keepdims=True)
        grad_wrt_f_max = tf.reduce_sum(
            dy * tf.where(above_max, ones, zeros),
            self.reduce_dims,
            keepdims=True)
      else:
        grad_wrt_f_min = tf.reduce_sum(dy * tf.where(below_min, ones, zeros))
        grad_wrt_f_max = tf.reduce_sum(dy * tf.where(above_max, ones, zeros))
      return grad_wrt_inputs, grad_wrt_f_min, grad_wrt_f_max

    return dequantized, grad_fn

  def call(self, inputs, f_min, f_max):
    """The fake quantization operation kernel.

    Args:
      inputs: a tensor containing values to be quantized.
      f_min: the minimum input value.
      f_max: the maximum input value.
    Returns:
      a tensor containing quantized values.
    """
    with tf.name_scope("FSFakeQuantize"):
      if self.use_framework_quant:
        if self.unsigned:
          return self.framework_fs_fake_quant_v2(inputs, f_min, f_max)
        else:
          return self.framework_fs_fake_quant(inputs, f_min, f_max)
      else:
        return self.custom_fs_fake_quant(inputs, f_min, f_max)


class Pof2SFakeQuantize(object):
  """The fake quantization with power-of-2 scale operation kernel."""

  def __init__(self,
               bit_width,
               method,
               round_mode,
               symmetry,
               per_channel,
               unsigned=False,
               narrow_range=False,
               reduce_dims=None):
    """Initialize the fake quantize with power-of-2 scale operation kernel.

    Args:
      bit_width: Number of bits to use for quantization, must be between 2 and 8.
      method: Integer Enum value of how to get the quantize pos, 0 for NON_OVERFLOW and 1 for MIN_MSE.
      round_mode: Int Enum value of the rounding mode. 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO.
      symmetry: Bool, whether to use symmetry quantization.
      per_channel: Bool, whether to use per-channel quantization.
      unsigned: Bool, whether to use unsigned integer for quantization.
      narrow_range: Bool, whether to use the narrow quantization range
        [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
      reduce_dims: List(Int) containing the dimensions to reduce, only take effect when per_channel=True.
    """
    self.bit_width = bit_width
    self.unsigned = unsigned
    self.narrow_range = False if self.unsigned else narrow_range
    if self.unsigned:
      bound = float(2**self.bit_width)
      self.q_min, self.q_max = 0., bound - 1
    else:
      bound = float(2**(self.bit_width - 1))
      self.q_min, self.q_max = -bound, bound - 1
      if self.narrow_range:
        self.q_min += 1

    self.method = QuantizeMethod(method)
    self.round_mode = vitis_round.RoundMode(round_mode)
    self.symmetry = symmetry
    self.per_channel = per_channel
    self.reduce_dims = reduce_dims

  @tf.custom_gradient
  def pof2s_fake_quant(self, inputs, quantize_pos, f_min, f_max):
    """The power-of-2 scale fake quantization operation kernel.

    Args:
      inputs: a tensor containing values to be quantized.
      quantize_pos: the quantize position.
      f_min: the minimum input value.
      f_max: the maximum input value.
    Returns:
      a tensor containing quantized values.
    """
    scale = tf.math.pow(2.0, quantize_pos, name="scale")
    if self.symmetry:
      shift = 0
    else:
      q_zero_point, new_f_min, new_f_max = quantize_zero_point(
          scale, f_min, f_max, self.q_min, self.q_max)
      shift = new_f_min

    quantized = quantize(inputs, scale, shift, self.q_min, self.q_max,
                         self.round_mode, self.symmetry)
    dequantized = dequantize(quantized, scale, shift, self.q_min, self.q_max,
                             self.round_mode, self.symmetry)

    def grad_fn(dy):
      """Custom gradient function."""
      return dy, None, None, None

    return dequantized, grad_fn

  def get_quantize_pos_non_overflow(self, inputs, f_min, f_max):
    """Get quantize pos which makes no value overflows."""
    with tf.name_scope("GetQuantizePosNonOverflow"):
      if self.symmetry:
        if self.unsigned:  # this condition will have f_min==self.q_min==0
          float_scale_inv = f_max / self.q_max
        else:
          min_scale_inv = f_min / self.q_min
          max_scale_inv = f_max / self.q_max
          float_scale_inv = tf.math.maximum(min_scale_inv, max_scale_inv)
      else:
        float_scale_inv = (f_max - f_min) / (self.q_max - self.q_min)
      # Avoid inf, using sys.float_info.epsilon, log2(epsilon) ~= 52
      float_scale_inv = tf.math.maximum(float_scale_inv, sys.float_info.epsilon)
      quantize_pos = -tf.math.log(float_scale_inv) / tf.math.log(2.0)
      quantize_pos = tf.math.floor(quantize_pos)
      return quantize_pos

  def get_quantize_pos_min_mse(self, inputs, f_min, f_max):
    """Get quantize pos which minimize mse between float and quantzed."""
    with tf.name_scope("GetQuantizePosMinMse"):
      non_overflow_pos = self.get_quantize_pos_non_overflow(
          inputs, f_min, f_max)

      mses = []
      for i in range(5):
        with tf.name_scope("FakeQuantizeWithScale_{}".format(i)):
          # fake quantize
          scale = tf.math.pow(2.0, non_overflow_pos + i, name="scale")
          if self.symmetry:
            shift = 0
          else:
            q_zero_point, new_f_min, new_f_max = quantize_zero_point(
                scale, f_min, f_max, self.q_min, self.q_max)
            shift = new_f_min
          quantized = quantize(inputs, scale, shift, self.q_min, self.q_max,
                               self.round_mode, self.symmetry)
          dequantized = dequantize(quantized, scale, shift, self.q_min,
                                   self.q_max, self.round_mode, self.symmetry)
          mse = tf.pow(inputs - dequantized, 2)
          mse = tf.reduce_sum(mse)
          mses.append(mse)
      pos_offset = tf.argmin(mses)
      quantize_pos = non_overflow_pos + tf.cast(pos_offset, tf.float32)
      return quantize_pos

  def get_quantize_pos(self, inputs, f_min, f_max):
    """Get the quantize position with given float range."""
    if self.method == QuantizeMethod.NON_OVERFLOW:
      return self.get_quantize_pos_non_overflow(inputs, f_min, f_max)
    elif self.method == QuantizeMethod.MIN_MSE:
      return self.get_quantize_pos_min_mse(inputs, f_min, f_max)
    else:
      logger.error('Invalid method: {}'.format(self.method))

  def call(self, inputs, quantize_pos, f_min, f_max):
    """The fake quantization operation kernel.

    Args:
      inputs: a tensor containing values to be quantized.
      quantize_pos: the quantize position.
      f_min: the minimum input value.
      f_max: the maximum input value.
    Returns:
      a tensor containing quantized values.
    """
    with tf.name_scope("Pof2SFakeQuantize"):
      return self.pof2s_fake_quant(inputs, quantize_pos, f_min, f_max)


class TQTFakeQuantize(object):
  """The fake quantization with trained quantization threshold operation kernel."""

  def __init__(self,
               bit_width,
               method,
               round_mode,
               symmetry,
               per_channel,
               unsigned=False,
               narrow_range=False,
               reduce_dims=None):
    """Initialize the fake quantize with thrained quantization threshold operation kernel.

    Args:
      bit_width: Number of bits to use for quantization, must be between 2 and 8.
      method: Int, method of how to get the quantize pos, 0 for NON_OVERFLOW and 1 for MIN_MSE.
      round_mode: Int Enum value of the rounding mode. 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO.
      symmetry: Bool, whether to use symmetry quantization.
      per_channel: Bool, whether to use per-channel quantization.
      unsigned: Bool, whether to use unsigned integer for quantization.
      narrow_range: Bool, whether to use the narrow quantization range
        [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
      reduce_dims: List(Int) containing the dimensions to reduce, only take effect when per_channel=True.
    """
    self.bit_width = bit_width
    self.unsigned = unsigned
    self.narrow_range = False if self.unsigned else narrow_range
    if self.unsigned:
      bound = float(2**self.bit_width)
      self.q_min, self.q_max = 0., bound - 1
    else:
      bound = float(2**(self.bit_width - 1))
      self.q_min, self.q_max = -bound, bound - 1
      if self.narrow_range:
        self.q_min += 1

    self.method = QuantizeMethod(method)
    self.round_mode = vitis_round.RoundMode(round_mode)
    self.symmetry = symmetry
    self.per_channel = per_channel
    # TODO: Support per_channel tqt quantize
    if self.per_channel:
      logger.error('Per_channel tqt quantize is not supported now.')
    self.reduce_dims = reduce_dims

  @tf.custom_gradient
  def tqt_fake_quant(self, inputs, log_th, f_min, f_max):
    """The trained quantization threshold fake quantization operation kernel.

    Args:
      inputs: a tensor containing values to be quantized.
      log_th: the log threshold.
      f_min: the minimum input value.
      f_max: the maximum input value.
    Returns:
      a tensor containing quantized values.
    """
    quantize_pos = self.bit_width - 1 - tf.math.ceil(log_th)
    scale = tf.math.pow(2.0, quantize_pos, name="scale")
    if self.symmetry:
      shift = 0
    else:
      q_zero_point, new_f_min, new_f_max = quantize_zero_point(
          scale, f_min, f_max, self.q_min, self.q_max)
      shift = new_f_min

    quantized = quantize(inputs, scale, shift, self.q_min, self.q_max,
                         self.round_mode, self.symmetry)
    dequantized = dequantize(quantized, scale, shift, self.q_min, self.q_max,
                             self.round_mode, self.symmetry)

    def grad_fn(dy):
      """Custom gradient function.

      grad_wrt_inputs = 1 if f_min < x < f_max else 0
                              [x * s] / s - x,  if q_min < [x * s] < q_max
      grad_wrt_log_th = ln2 * q_min / s,        if [x * s] < q_min
                              q_max / s,        if [x * s] > q_max
      """
      if self.symmetry:
        _f_min, _f_max = f_min, f_max
      else:
        _f_min, _f_max = new_f_min, new_f_max

      between_min_max = (inputs >= _f_min) & (inputs <= _f_max)
      ones = tf.ones_like(dy)
      zeros = tf.zeros_like(dy)
      grad_wrt_inputs = dy * tf.where(between_min_max, ones, zeros)
      grad_wrt_log_th = tf.reduce_sum(
          dy * tf.math.log(2.0) *
          tf.where(between_min_max, dequantized - inputs, quantized / scale))

      return grad_wrt_inputs, grad_wrt_log_th, None, None

    return dequantized, grad_fn

  def get_log_th_non_overflow(self, inputs, f_min, f_max):
    """Get log threshold which makes no value overflows."""
    with tf.name_scope("GetLogThNonOverflow"):
      f_min_abs = tf.math.abs(f_min)
      f_max_adj = f_max * (-self.q_min / self.q_max)
      th = tf.math.maximum(f_min_abs, f_max_adj)
      th = tf.math.maximum(th, 1e-9)
      return tf.math.divide(tf.math.log(th), tf.math.log(2.))

  def get_log_th(self, inputs, f_min, f_max):
    """Get the log threshold with given float range."""
    if self.method == QuantizeMethod.NON_OVERFLOW:
      return self.get_log_th_non_overflow(inputs, f_min, f_max)
    else:
      logger.error('Invalid method: {}'.format(self.method))

  def call(self, inputs, log_th, f_min, f_max):
    """The fake quantization operation kernel.

    Args:
      inputs: a tensor containing values to be quantized.
      log_th: the log threshold.
      f_min: the minimum input value.
      f_max: the maximum input value.
    Returns:
      a tensor containing quantized values.
    """
    with tf.name_scope("TQTFakeQuantize"):
      return self.tqt_fake_quant(inputs, log_th, f_min, f_max)


def convert_channel_axis_to_reduce_dims(input_dims, channel_axis):
  """Helper function to convert channel_axis to reduce_dims."""
  if channel_axis < 0:
    channel_axis = input_dims + channel_axis
  reduce_dims = [i for i in range(input_dims) if i != channel_axis]
  return reduce_dims


def convert_reduce_dims_to_channel_axis(input_dims, reduce_dims):
  """Helper function to convert reduce_dims to channel_axis."""
  channel_axis = [i for i in range(input_dims) if i not in reduce_dims]
  return channel_axis[0]


def FSQuantize(
    inputs,
    min_var,
    max_var,
    calib_hist,
    calib_bin_edges,
    bit_width,
    method,
    round_mode,
    mode,
    is_training,
    symmetry,
    per_channel,
    channel_axis,
    use_framework_quant=True,
    unsigned=False,
    narrow_range=False,
    name_scope="FSQuantize",
):
  """Float scale quantize op.

  Args:
    inputs: Input values.
    min_var: Variable of minimum value of inputs.
    max_var: Variable of maximum value of inputs.
    calib_hist: Variable of histogram of inputs. 
    calib_bin_edges: Variable of linspace of inputs.
    bit_width: Int, bit width of quantized values.
    method: method of quantize valued of inputs,
    round_mode: Int, the mode of rounding function, 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO.
    mode: String, the mode of quantization, available modes are ['ANALYSE', 'QCB', 'QCBEV', 'QAT']
    is_training: Bool, whether in training phase.
    symmetry: Bool, whether to apply symmetry quantization.
    per_channel: Bool, whether to apply per_channel quantization.
    channel_axis: The axis of the channel, used with per_channel enabled. The last dimension is 
      regarded as channel axis and other dimension will be reduces by default.
    use_framework_quant: Bool, whether to use the tensorflow fake_quantize operations. If not, the custom
      quantize kernel will be used.
    unsigned: Bool, whether to use unsigned integer for quantization.
    narrow_range: Bool, whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].

  Return:
    Quantized inputs.
  """
  with tf.name_scope(name_scope):
    reduce_dims = None
    if per_channel:
      input_dims = len(inputs.get_shape())
      reduce_dims = convert_channel_axis_to_reduce_dims(input_dims,
                                                        channel_axis)

    quantize_kernel = FSFakeQuantize(
        bit_width=bit_width,
        round_mode=round_mode,
        symmetry=symmetry,
        per_channel=per_channel,
        use_framework_quant=use_framework_quant,
        unsigned=unsigned,
        narrow_range=narrow_range,
        reduce_dims=reduce_dims)

    # ANALYSE branch
    if mode == 'ANALYSE':
      batch_min, batch_max = get_min_max(
          inputs,
          bit_width,
          method,
          symmetry=symmetry,
          per_channel=per_channel,
          unsigned=unsigned,
          narrow_range=narrow_range,
          reduce_dims=reduce_dims)
      assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
      assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')
      return tf.identity(inputs, name='identity')

    if is_training or mode == 'QCB':
      # Training and calibration branch
      batch_min = None
      batch_max = None
      method = QuantizeMethod(method)
      if method == QuantizeMethod.NON_OVERFLOW or method == QuantizeMethod.MIN_MSE:
        batch_min, batch_max = get_min_max(
            inputs,
            bit_width,
            method,
            symmetry=symmetry,
            per_channel=per_channel,
            unsigned=unsigned,
            narrow_range=narrow_range,
            reduce_dims=reduce_dims)
        #if not per_channel:
        batch_min = tf.math.minimum(min_var, batch_min)
        batch_max = tf.math.maximum(max_var, batch_max)
        assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
        assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')
        return quantize_kernel.call(inputs, assign_min, assign_max)

      elif method == QuantizeMethod.MIN_KL:
        _calib_hist, _calib_bin_edges = calibrator_numpy.numpy_collect(
            inputs, calib_hist, calib_bin_edges)
        calib_hist = tf_compat.assign(
            calib_hist, _calib_hist, name='calib_hist')
        calib_bin_edges = tf_compat.assign(
            calib_bin_edges, _calib_bin_edges, name='calib_bin_edges')
        return tf.identity(inputs, name='identity')

      elif method == QuantizeMethod.PERCENTILE:
        _calib_hist, _calib_bin_edges = calibrator_numpy.numpy_collect(
            inputs, calib_hist, calib_bin_edges)
        calib_hist = tf_compat.assign(
            calib_hist, _calib_hist, name='calib_hist')
        calib_bin_edges = tf_compat.assign(
            calib_bin_edges, _calib_bin_edges, name='calib_bin_edges')
        return tf.identity(inputs, name='identity')
      else:
        logger.error('Invalid method: {}'.format(method))
        return tf.identity(inputs, name='identity')

    else:
      # Evaluation branch
      return quantize_kernel.call(inputs, min_var, max_var)


def MAFSQuantize(inputs,
                 min_var,
                 max_var,
                 bit_width,
                 round_mode,
                 mode,
                 is_training,
                 symmetry,
                 per_channel,
                 channel_axis,
                 ema_decay=0.999,
                 use_framework_quant=True,
                 unsigned=False,
                 narrow_range=False,
                 name_scope="MAFSQuantize"):
  """Moving average float scale quantize op.

  Args:
    inputs: Input values.
    min_var: Variable of minimum value of inputs.
    max_var: Variable of maximum value of inputs.
    bit_width: Int, bit width of quantized values.
    round_mode: Int, the mode of rounding function, 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO.
    mode: String, the mode of quantization, available modes are ['ANALYSE', 'QCB', 'QCBEV', 'QAT']
    is_training: Bool, whether in training phase.
    symmetry: Bool, whether to apply symmetry quantization.
    per_channel: Bool, whether to apply per_channel quantization.
    channel_axis: The axis of the channel, used with per_channel enabled. The last dimension is 
      regarded as channel axis and other dimension will be reduces by default.
    ema_decay: Float, EMA decay parameter.
    use_framework_quant: Bool, whether to use the tensorflow fake_quantize operations. If not, the custom
      quantize kernel will be used.
    unsigned: Bool, whether to use unsigned integer for quantization.
    narrow_range: Bool, whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].

  Return:
    Quantized inputs.
  """
  with tf.name_scope(name_scope):
    reduce_dims = None
    if per_channel:
      input_dims = len(inputs.get_shape())
      reduce_dims = convert_channel_axis_to_reduce_dims(input_dims,
                                                        channel_axis)

    quantize_kernel = MAFSFakeQuantize(
        bit_width=bit_width,
        round_mode=round_mode,
        symmetry=symmetry,
        per_channel=per_channel,
        use_framework_quant=use_framework_quant,
        unsigned=unsigned,
        narrow_range=narrow_range,
        reduce_dims=reduce_dims)

    # ANALYSE branch
    if mode == 'ANALYSE':
      batch_min, batch_max = get_min_max(
          inputs,
          bit_width,
          symmetry=symmetry,
          per_channel=per_channel,
          unsigned=unsigned,
          narrow_range=narrow_range,
          reduce_dims=reduce_dims)
      assign_min = moving_averages.assign_moving_average(
          min_var,
          batch_min,
          ema_decay,
          zero_debias=False,
          name='assign_min_ema')
      assign_max = moving_averages.assign_moving_average(
          max_var,
          batch_max,
          ema_decay,
          zero_debias=False,
          name='assign_max_ema')
      return tf.identity(inputs, name='identity')

    if is_training or mode == 'QCB':
      # Training and calibration branch
      batch_min, batch_max = get_min_max(
          inputs,
          bit_width,
          symmetry=symmetry,
          per_channel=per_channel,
          unsigned=unsigned,
          narrow_range=narrow_range,
          reduce_dims=reduce_dims)
      assign_min = moving_averages.assign_moving_average(
          min_var,
          batch_min,
          ema_decay,
          zero_debias=True,
          name='assign_min_ema')
      assign_max = moving_averages.assign_moving_average(
          max_var,
          batch_max,
          ema_decay,
          zero_debias=True,
          name='assign_max_ema')
      return quantize_kernel.call(inputs, assign_min, assign_max)
    else:
      # Evaluation branch
      return quantize_kernel.call(inputs, min_var, max_var)


def Pof2SQuantize(inputs,
                  quant_pos_var,
                  min_var,
                  max_var,
                  bit_width,
                  method,
                  round_mode,
                  mode,
                  is_training,
                  symmetry,
                  per_channel,
                  channel_axis,
                  unsigned=False,
                  narrow_range=False,
                  name_scope="Pof2SQuantize"):
  """Power-of-2 quantize op with quantize position. 

  Args:
    inputs: Input values.
    quant_pos_var: Variable of quantize position.
    min_var: Variable of minimum value of inputs.
    max_var: Variable of maximum value of inputs.
    bit_width: Int, bit width of quantized values.
    method: Int Enum, method of how to get the quantize pos, 0 for NON_OVERFLOW and 1 for MIN_MSE.
    round_mode: Int, the mode of rounding function, 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO.
      By default, weights are quantized with HALF_TO_EVEN round mode, inputs and activations are quantized with HALF_UP round mode.
    mode: String, the mode of quantization, available modes are ['ANALYSE', 'QCB', 'QCBEV', 'QAT']
    is_training: Bool, whether in training phase.
    symmetry: Bool, whether to apply symmetry quantization.
    per_channel: Bool, whether to apply per_channel quantization.
    channel_axis: The axis of the channel, used with per_channel enabled. The last dimension is 
      regarded as channel axis and other dimension will be reduces by default.
    unsigned: Bool, whether to use unsigned integer for quantization.
    narrow_range: Bool, whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].

  Return:
    Quantized inputs.
  """
  with tf.name_scope(name_scope):
    reduce_dims = None
    if per_channel:
      input_dims = len(inputs.get_shape())
      reduce_dims = convert_channel_axis_to_reduce_dims(input_dims,
                                                        channel_axis)

    quantize_kernel = Pof2SFakeQuantize(
        bit_width=bit_width,
        method=method,
        round_mode=round_mode,
        symmetry=symmetry,
        per_channel=per_channel,
        unsigned=unsigned,
        narrow_range=narrow_range,
        reduce_dims=reduce_dims)

    # ANALYSE branch
    if mode == 'ANALYSE':
      batch_min, batch_max = get_min_max(
          inputs,
          bit_width,
          symmetry=symmetry,
          per_channel=per_channel,
          unsigned=unsigned,
          narrow_range=narrow_range,
          reduce_dims=reduce_dims)
      assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
      assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')
      return tf.identity(inputs, name='identity')

    if is_training or mode == 'QCB':
      # Training and calibration branch
      batch_min, batch_max = get_min_max(
          inputs,
          bit_width,
          symmetry=symmetry,
          per_channel=per_channel,
          unsigned=unsigned,
          narrow_range=narrow_range,
          reduce_dims=reduce_dims)
      assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
      assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')

      # Get quantize positions
      batch_quantize_pos = quantize_kernel.get_quantize_pos(
          inputs, assign_min, assign_max)
      assign_quantize_pos = tf_compat.assign(
          quant_pos_var, batch_quantize_pos, name="assign_quantize_pos")
      return quantize_kernel.call(inputs, assign_quantize_pos, assign_min,
                                  assign_max)
    else:
      # Evaluation branch
      return quantize_kernel.call(inputs, quant_pos_var, min_var, max_var)


def TQTQuantize(inputs,
                log_th_var,
                min_var,
                max_var,
                bit_width,
                method,
                round_mode,
                mode,
                is_training,
                symmetry,
                per_channel,
                channel_axis,
                unsigned=False,
                narrow_range=False,
                name_scope="TQTQuantize"):
  """Power-of-2 quantize op with log threshold.

  Args:
    inputs: Input values.
    log_th_var: Variable of log threshold.
    min_var: Variable of minimum value of inputs.
    max_var: Variable of maximum value of inputs.
    bit_width: Int, bit width of quantized values.
    method: Int Enum, method of how to get the initial log threshold, 0 for non_overflow.
    round_mode: Int, the mode of rounding function, 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO.
    mode: String, the mode of quantization, available modes are ['ANALYSE', 'QCB', 'QCBEV', 'QAT']
    is_training: Bool, whether in training phase.
    symmetry: Bool, whether to apply symmetry quantization.
    per_channel: Bool, whether to apply per_channel quantization.
    channel_axis: The axis of the channel, used with per_channel enabled. The last dimension is 
      regarded as channel axis and other dimension will be reduces by default.
    unsigned: Bool, whether to use unsigned integer for quantization.
    narrow_range: Bool, whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].

  Return:
    Quantized inputs.
  """
  with tf.name_scope(name_scope):
    reduce_dims = None
    if per_channel:
      input_dims = len(inputs.get_shape())
      reduce_dims = convert_channel_axis_to_reduce_dims(input_dims,
                                                        channel_axis)

    quantize_kernel = TQTFakeQuantize(
        bit_width=bit_width,
        method=method,
        round_mode=round_mode,
        symmetry=symmetry,
        per_channel=per_channel,
        unsigned=unsigned,
        narrow_range=narrow_range,
        reduce_dims=reduce_dims)

    # ANALYSE branch
    if mode == 'ANALYSE':
      batch_min, batch_max = get_min_max(
          inputs,
          bit_width,
          symmetry=symmetry,
          per_channel=per_channel,
          unsigned=unsigned,
          narrow_range=narrow_range,
          reduce_dims=reduce_dims)
      assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
      assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')
      return tf.identity(inputs, name='identity')

    if is_training or mode == 'QCB':
      # Training and calibration branch
      batch_min, batch_max = get_min_max(
          inputs,
          bit_width,
          symmetry=symmetry,
          per_channel=per_channel,
          unsigned=unsigned,
          narrow_range=narrow_range,
          reduce_dims=reduce_dims)
      assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
      assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')

      if mode == 'QCB':
        batch_log_th = quantize_kernel.get_log_th(inputs, assign_min,
                                                  assign_max)
        assign_log_th = tf_compat.assign(
            log_th_var, batch_log_th, name="assign_log_th")
        return quantize_kernel.call(inputs, assign_log_th, assign_min,
                                    assign_max)
      else:
        return quantize_kernel.call(inputs, log_th_var, assign_min, assign_max)
    else:
      # Evaluation branch
      return quantize_kernel.call(inputs, log_th_var, min_var, max_var)
