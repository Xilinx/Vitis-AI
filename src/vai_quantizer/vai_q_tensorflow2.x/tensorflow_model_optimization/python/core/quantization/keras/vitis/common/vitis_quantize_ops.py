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

import os
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

def get_perm(n_dim, axis):
  perm = [i for i in range(n_dim)]
  perm[-1] = perm[axis]
  perm[axis] = n_dim - 1
  return perm

def sym_quantize(inputs, scale, q_min, q_max, round_mode):
  """Symmetry Quantize Kernel.  Q(x) = q_min + round[(x * scale]."""
  with tf.name_scope("SymQuantize"):
    inputs = tf.cast(inputs, tf.float32)
    rounded = vitis_round.round(inputs * scale, round_mode)
    quantized = tf.clip_by_value(rounded, q_min, q_max)
  return quantized


def asym_quantize(inputs, scale, shift, q_min, q_max, round_mode):
  """Asymmetry Quantize Kernel.  Q(x) = q_min + round[(x - shift) * scale]."""
  with tf.name_scope("AsymQuantize"):
    inputs = tf.cast(inputs, tf.float32)
    rounded = vitis_round.round((inputs - shift) * scale, round_mode)
    quantized = tf.clip_by_value(q_min + rounded, q_min, q_max)
  return quantized


def quantize(inputs, scale, shift, q_min, q_max, round_mode, symmetry):
  """Quantize Kernel."""
  if symmetry:
    return sym_quantize(inputs, scale, q_min, q_max, round_mode)
  else:
    return asym_quantize(inputs, scale, shift, q_min, q_max, round_mode)

def bfp_quantize(inputs, scale, round_mode, min_v, max_v):
  with tf.name_scope("BFPQuantize"):
    rounded = vitis_round.round(inputs/scale, round_mode) * scale
    quantized = tf.clip_by_value(rounded, min_v, max_v)
  return quantized

def bfp_dequantize(inputs, shape, axis, tmp_shape):
  """Transform block-wised tensor to given `shape` and remove padded blocks if it has.
      For example, given a block-wised tensor with shape [N*W*H, L, B],
      shape=[N, H, W, C] and axis=3. The function first reshape it to [N*W*H, L*B] 
      and remove padded blocks to [N*W*H, C], then transform it to given shape [N, H, W, C].

      Args:
        inputs: A tensor in block-wised format.
        shape: The shape of the output tensor.
        axis: The axis where the channels is located in the `shape`.
        tmp_shape: The shape of the origin tensor that the channel dimension has been transformed to last dimension.

      Returns:
        A transposed and de-padded tensor with given `shape`.
  """
  with tf.name_scope("BFPDequantize"):
    origin_n_dim = shape.shape[0]
    _, L, B = inputs.shape
    # [N*W*H, L, B] -> [N*H*W, L*B] -> [N*W*H, C]
    inputs = tf.reshape(inputs, [-1, L*B])
    inputs = tf.slice(inputs, [0,0], [-1, shape[axis]])
    if axis == -1 or axis == origin_n_dim-1: 
      dequantized = tf.reshape(inputs, shape)
    else:
      perm = get_perm(origin_n_dim, axis)
      dequantized = tf.reshape(inputs, tmp_shape)
      dequantized = tf.transpose(dequantized, perm=perm)
  return dequantized


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
      batch_max = tf.cast(batch_max, tf.float32)
      batch_min = tf.cast(batch_min, tf.float32)
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
               use_fixneuron_quant=0,
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
      use_fixneuron_quant: Int, 0 for not to use fixneuron (the general quantize kernel will be used),
        1 for using fixneuron to quantize activation, 2 for using fixneuron to quantize weights.
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
    self.use_fixneuron_quant = use_fixneuron_quant

    # fixneuron only supports limited settings
    if self.use_fixneuron_quant:
      if self.bit_width != 8 or self.unsigned or self.narrow_range or \
         (self.method != QuantizeMethod.NON_OVERFLOW and \
          self.method != QuantizeMethod.MIN_MSE) or \
         self.symmetry == False or self.per_channel:
        logger.warning('Disable fixneuron because of unsupported settings:'
          'bit_width {}, unsigned {}, narrow_range {}, method {}, '
          'symmetry {}, per_channel {}'.format(self.bit_width, self.unsigned,
          self.narrow_range, self.method, self.symmetry, self.per_channel)
        )
        self.use_fixneuron_quant = 0

    # Prepare output directory for fixneuron
    self.fixneuron_output_dir = './'
    if self.use_fixneuron_quant:
      temp_path = os.path.join(self.fixneuron_output_dir, "temp")
      if not os.path.exists(temp_path):
        os.makedirs(temp_path)

  def fixneuron_pof2s_fake_quant(self, inputs, quantize_pos, quantize_phase):
    """The power-of-2 scale fake quantization by fixneuron.

    Args:
      inputs: a tensor containing values to be quantized.
      quantize_pos: the quantize position.
      quantize_phase: the quantize phase. 0: Calibration 1: Evaluation, 2: Training
    Returns:
      a tensor containing quantized values.
    """
    fixneuron_bit_width = self.bit_width
    fixneuron_method = 0 if self.method == QuantizeMethod.NON_OVERFLOW else 1

    fixneuron_mode = 1 if self.use_fixneuron_quant == 2 else 0
    fixneuron_quantize_pos = tf.keras.backend.get_value(quantize_pos)

    from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_q_tensorflow import FixNeuron
    return FixNeuron(input=inputs, bit_width=fixneuron_bit_width,
                     method=fixneuron_method,
                     mode=fixneuron_mode, phase=quantize_phase,
                     output_dir=self.fixneuron_output_dir,
                     quantize_pos=fixneuron_quantize_pos)

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

  def call(self, inputs, quantize_pos, quantize_phase, f_min, f_max):
    """The fake quantization operation kernel.

    Args:
      inputs: a tensor containing values to be quantized.
      quantize_pos: the quantize position.
      quantize_phase: the quantize phase (used by fixneuron).
      f_min: the minimum input value.
      f_max: the maximum input value.
    Returns:
      a tensor containing quantized values.
    """
    with tf.name_scope("Pof2SFakeQuantize"):
      if self.use_fixneuron_quant:
        return self.fixneuron_pof2s_fake_quant(inputs, quantize_pos, quantize_phase)
      else:
        return self.pof2s_fake_quant(inputs, quantize_pos, f_min, f_max)


class TQTFakeQuantize(object):
  """The fake quantization with trained quantization threshold operation kernel."""

  def __init__(self,
               bit_width,
               method,
               round_mode,
               symmetry,
               per_channel,
               use_fixneuron_quant=0,
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
      use_fixneuron_quant: Int, 0 for not to use fixneuron (the general quantize kernel will be used),
        1 for using fixneuron to quantize activation, 2 for using fixneuron to quantize weights.
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
    self.use_fixneuron_quant = use_fixneuron_quant

    # fixneuron does not support tqt quantization
    if self.use_fixneuron_quant:
      logger.warning('The tqt quantize does not support using fixneuron.')

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
      scaled = inputs * scale
      rounded = vitis_round.round(scaled, self.round_mode)
      between_min_max = (rounded >= self.q_min) & (rounded <= self.q_max)

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

    if (is_training and mode == 'QAT') or mode == 'QCB':
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

    if (is_training and mode == 'QAT') or mode == 'QCB':
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
                  use_fixneuron_quant=0,
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
    use_fixneuron_quant: Int, 0 for not to use fixneuron (the general quantize kernel will be used),
      1 for using fixneuron to quantize activation, 2 for using fixneuron to quantize weights.
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
    quantize_phase = 1
    if use_fixneuron_quant:
      if is_training:
        quantize_phase = 2
      elif mode == 'QCB':
        quantize_phase = 0
      else:  # mode == 'ANALYSE' or 'QAT'
        quantize_phase = 1

    quantize_kernel = Pof2SFakeQuantize(
        bit_width=bit_width,
        method=method,
        round_mode=round_mode,
        symmetry=symmetry,
        per_channel=per_channel,
        use_fixneuron_quant=use_fixneuron_quant,
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

    if (is_training and mode == 'QAT') or mode == 'QCB':
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
      return quantize_kernel.call(inputs, assign_quantize_pos, quantize_phase,
                                  assign_min, assign_max)
    else:
      # Evaluation branch
      return quantize_kernel.call(inputs, quant_pos_var, quantize_phase,
                                  min_var, max_var)


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
                use_fixneuron_quant=0,
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
    use_fixneuron_quant: Int, 0 for not to use fixneuron (the general quantize kernel will be used),
      1 for using fixneuron to quantize activation, 2 for using fixneuron to quantize weights.
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
        use_fixneuron_quant=use_fixneuron_quant,
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

    if (is_training and mode == 'QAT') or mode == 'QCB':
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


def BFPQuantize(
    inputs,
    tensor_shape,
    data_format,
    bit_width,
    round_mode,
    axis,
    tile_size,
    is_training,
    name_scope="BFPQuantize",
):
  """Float scale quantize op.

  Args:
    inputs: Input values.
    data_format: "bfp"/"bf16"/"fp32"
    bit_width: Number of bits to use for bfp, must be bigger than 8.
    axis: The axis where the channels is located in the `shape`.
    tile_size: The number of tensors in a block.
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
    narrow_range: Bool, whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].

  Return:
    Quantized inputs.
  """
  with tf.name_scope(name_scope):

    quantize_kernel = BFPFakeQuantize(
        tensor_shape=tensor_shape,
        data_format=data_format,
        bit_width=bit_width,
        round_mode=round_mode,
        axis=axis,
        tile_size=tile_size,
        is_training=is_training)

    return quantize_kernel.call(inputs)

class BFPFakeQuantize(object):
  """The fake quantization with dataformat operation kernel."""

  def __init__(self,
               tensor_shape=None,
               data_format="bfp",
               bit_width=16,
               round_mode=0,
               axis=-1,
               tile_size=8,
               epsilon=tf.math.pow(tf.constant(2.0), -23),
               is_training=False):
    """Initialize the fake quantize with dataformat operation kernel.
    For "MSFP" data format, block_size=16, exponent_bits=8, sub-block_size=2, sub-block_shift_bits=1, mantissa_bits=bit_width-exponent_bits-sub_block_shift_bits.

    Args:
      data_format: "bfp"/"bf16"/"fp32"/"msfp"
      bit_width: Number of bits to use for bfp, must be bigger than 8. Or number of bits to use for msfp, must be 16/13/11.
      round_mode: Int Enum value of the rounding mode. 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO.
      axis: The axis where the channels is located in the `shape`.
      tile_size: The number of tensors in a block.
      is_training: Bool, whether in training phase.
    """
    self.tensor_shape = tensor_shape
    self.data_format = data_format
    self.bit_width = bit_width
    self.round_mode = vitis_round.RoundMode(round_mode)
    self.axis = axis
    self.tile_size = tile_size
    self.is_training = is_training
    self.epsilon = epsilon
    if self.data_format == 'msfp':
      assert bit_width in [16, 13, 11], 'Number of bits used for MSFP must be 16/13/11.'
      self.block_size = 12
      sub_block_size = 2
      self.tile_size = sub_block_size
      self.round_mode = vitis_round.RoundMode(2)

  def _transform_to_block_wise(self, inputs):
    """Transform input tensor to block-wised format (i.e. the block in last dimension) at given asix with paddings if needed.

    For example, given a input tensor with shape [N, C, H, W], axis=1.
    The tensor will be transposed to [N*W*H, L, B], where B equals to `tile_size`. 
    The channels will be padded with zeros before transpose to make it divisble by `tile_size`.

    Args:
      inputs: Input tensor.

    Returns:
      A transposed and padded tensor in block-wised format.
    """

    tmp_shape = None
    #C = inputs.shape[self.axis]
    #n_dim = inputs.shape.ndims
    C = self.tensor_shape[self.axis]
    n_dim = self.tensor_shape.ndims
    assert np.abs(self.axis) <= n_dim 
    if self.axis != -1 and self.axis != n_dim-1:
      perm = get_perm(n_dim, self.axis)
      inputs = tf.transpose(inputs, perm=perm)
      tmp_shape = tf.shape(inputs)
    # [N, W, H, C] -> [N*W*H, C]
    inputs = tf.reshape(inputs, [-1, C])
    padded_channels = self.tile_size - C % self.tile_size
    if padded_channels != self.tile_size:
      inputs = tf.pad(inputs, tf.constant([[0,0,], [0,padded_channels]]), "CONSTANT")
    # [N*W*H, C] -> [N*W*H, L, B]
    #_, C = inputs.shape
    #return tf.reshape(inputs, [-1, int(C/self.tile_size), self.tile_size]), tmp_shape
    C = tf.shape(inputs)[-1]
    return tf.reshape(inputs, [-1, tf.divide(C, self.tile_size), self.tile_size]), tmp_shape

  def _get_exponent(self, inputs):
    t = tf.abs(inputs)
    # use fp32's 1.mantissa_bits
    max_t = tf.math.reduce_max(t, axis=-1, keepdims=True)
    max_exp = tf.math.floor(tf.math.log(max_t + self.epsilon) / tf.math.log(2.0))
    t_exp = tf.math.floor(tf.math.log(t + self.epsilon) / tf.math.log(2.0))
    return max_exp, t_exp

  def _get_shared_exponent(self, sub_max_exp):
    # sub_max_exp:  maximum exponent in one sub-block
    n = int(self.block_size / self.tile_size)
    C = sub_max_exp.shape[-2]
    t = tf.squeeze(sub_max_exp, [-1])
    padded_channels = n - C % n
    if padded_channels != n:
      t = tf.pad(t, tf.constant([[0,0,], [0,padded_channels]]), "CONSTANT", constant_values=tf.float32.min)
      C = t.shape[-1]
    convert_t = tf.reshape(t, [-1,int(C/n),n])
    convert_t = tf.tile(convert_t, [1,1,n])
    convert_t = tf.reshape(convert_t, [-1,C,n])
    # shared_max_exp: maximum expoment in one block
    shared_max_exp = tf.math.reduce_max(convert_t, axis=-1, keepdims=True)
    if padded_channels != n:
      shape = tf.shape(shared_max_exp)
      shared_max_exp = tf.slice(shared_max_exp, [0,0,0], [shape[0], shape[1]-padded_channels, 1])      
    return shared_max_exp

  def _get_exponent_with_shift(self, shared_exp, sub_exp):
    # d: number of bits in the MSFP' sub-block shift field
    d = 1
    threshold = tf.cast(tf.pow(2, d) - 1, tf.float32)
    less = tf.less(tf.subtract(shared_exp, sub_exp), threshold)
    max_exp = tf.where(less, sub_exp, tf.subtract(shared_exp, threshold))
    return max_exp

  def _get_smallest_and_largest(self, exp):
    # sign bits: 1, exponent bits: 8, no implicit leading 1
    mantissa_bits = self.bit_width - 9
    # The min/max representable value with exp
    smallest = tf.math.pow(2.0, exp - (mantissa_bits - 1))
    largest = tf.math.pow(2.0, exp + 1) - smallest
    return smallest, largest

  def _get_smallest_and_largest_shared(self, exp, shared_exp):
    # sign bits: 1, exponent bits: 8, no implicit leading 1
    mantissa_bits = self.bit_width - 9
    # The min/max representable value with exp
    smallest = tf.math.pow(2.0, exp - (mantissa_bits - 1))
    largest = tf.math.pow(2.0, shared_exp + 1) - smallest
    return smallest, largest

  @tf.custom_gradient
  def custom_bfp_fake_quant(self, inputs):
    """The custom dataformat-bfp fake quantization operation kernel.

    Args:
      inputs: a tensor containing values to be quantized.
    Returns:
      a tensor containing quantized values.
    """
    # Handle inf/nan value in the input tensor.
    inf_mask = tf.cast(tf.math.is_inf(inputs), tf.float32)
    inf_remain = tf.math.multiply(inputs, inf_mask)
    inputs = tf.math.multiply_no_nan(inputs,tf.cast(tf.math.logical_not(tf.math.logical_or(tf.math.is_inf(inputs), tf.math.is_nan(inputs))),tf.float32))

    input_shape = tf.shape(inputs)
    inputs, tmp_shape = self._transform_to_block_wise(inputs)
    max_exp, _ = self._get_exponent(inputs)
    interval, max_v = self._get_smallest_and_largest(max_exp)
    quantized = bfp_quantize(inputs, interval, self.round_mode, -max_v, max_v)
    dequantized = bfp_dequantize(quantized, input_shape, self.axis, tmp_shape)

    dequantized = tf.math.add(dequantized, inf_remain)


    def grad_fn(dy):
      """Custom gradient function."""
      return dy 

    return dequantized, grad_fn

  @tf.custom_gradient
  def custom_msfp_fake_quant(self, inputs):
    """The custom dataformat-bfp fake quantization operation kernel.

    Args:
      inputs: a tensor containing values to be quantized.
    Returns:
      a tensor containing quantized values.
    """
    # Handle inf/nan value in the input tensor.
    inf_mask = tf.cast(tf.math.is_inf(inputs), tf.float32)
    inf_remain = tf.math.multiply(inputs, inf_mask)
    inputs = tf.math.multiply_no_nan(inputs,tf.cast(tf.math.logical_not(tf.math.logical_or(tf.math.is_inf(inputs), tf.math.is_nan(inputs))),tf.float32))

    input_shape = tf.shape(inputs)
    inputs, tmp_shape = self._transform_to_block_wise(inputs)
    sub_max_exp, _ = self._get_exponent(inputs)
    shared_max_exp = self._get_shared_exponent(sub_max_exp)
    max_exp = self._get_exponent_with_shift(shared_max_exp, sub_max_exp)

    interval, max_v = self._get_smallest_and_largest_shared(max_exp, shared_max_exp)
    quantized = bfp_quantize(inputs, interval, self.round_mode, -max_v, max_v)
    dequantized = bfp_dequantize(quantized, input_shape, self.axis, tmp_shape)

    dequantized = tf.math.add(dequantized, inf_remain)


    def grad_fn(dy):
      """Custom gradient function."""
      return dy 

    return dequantized, grad_fn

  @tf.custom_gradient
  def custom_bf16_fake_quant(self, inputs):
    """The custom dataformat-bf16 fake quantization operation kernel.

    Args:
      inputs: a tensor containing values to be quantized.
    Returns:
      a tensor containing quantized values.
    """
    quantized = tf.cast(inputs, tf.bfloat16)
    dequantized = tf.cast(quantized, tf.float32)
    def grad_fn(dy):
      """Custom gradient function."""
      return dy 

    return dequantized, grad_fn

  def call(self, inputs):
    """The fake dataformat quantization operation kernel.

    Args:
      inputs: a tensor containing values to be quantized.
    Returns:
      a tensor containing quantized values.
    """
    if self.data_format == "bfp":
      with tf.name_scope("BFP_BFPFakeQuantize"):
        return self.custom_bfp_fake_quant(inputs)
    elif self.data_format == "msfp":
      with tf.name_scope("BFP_MSFPFakeQuantize"):
        return self.custom_msfp_fake_quant(inputs)
    elif self.data_format == "bf16":
      with tf.name_scope("BFP_BF16FakeQuantize"):
        return self.custom_bf16_fake_quant(inputs)
    elif self.data_format == "fp32":
      with tf.name_scope("BFP_FP32FakeQuantize"):
        return inputs
    else:
      pass

