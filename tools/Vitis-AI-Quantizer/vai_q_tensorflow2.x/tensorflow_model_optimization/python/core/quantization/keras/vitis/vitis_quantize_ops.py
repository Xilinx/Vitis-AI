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
"""Python support for quantization operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.training import moving_averages
from tensorflow_model_optimization.python.core.keras import compat as tf_compat
from tensorflow.keras import layers


def quantize(inputs, scale, shift, q_min, q_max):
  """Quantize Kernel.  Q(x) = q_min + round[(x-shift) * scale]. """
  with tf.name_scope("Quantize"):
    rounded = tf.math.round((inputs - shift) * scale)
    quantized = tf.clip_by_value(q_min + rounded, q_min, q_max)
  return quantized


def dequantize(inputs, scale, shift, q_min, q_max):
  """Dequantize Kernel.  DQ(x) =  (x - q_min) / scale + shift. """
  with tf.name_scope("Dequantize"):
    return (inputs - q_min) / scale + shift


def symmetry_quantize(inputs, scale, q_min, q_max):
  """Quantize Kernel.  Q(x) = round[(x) * scale]. """
  with tf.name_scope("Quantize"):
    rounded = tf.math.round(inputs * scale)
    quantized = tf.clip_by_value(rounded, q_min, q_max)
  return quantized


def symmetry_dequantize(inputs, scale, q_min, q_max):
  """Dequantize Kernel.  DQ(x) =  x / scale. """
  with tf.name_scope("Dequantize"):
    return inputs / scale


def dpu_symmetry_quantize(inputs, scale, q_min, q_max):
  """DPU Quantize Kernel.  Q(x) = dpu_round[(x) * scale]. 
    
    dpu_round(x)  =   round(x)  if x - floor(x) != 0.5
                  =   ceil(x)   if x - floor(x) == 0.5
  """
  with tf.name_scope("DPUQuantize"):
    scaled = inputs * scale
    floored = tf.math.floor(scaled)
    ceiled = tf.math.ceil(scaled)
    rounded = tf.math.round(scaled)
    dpu_rounded = tf.where(
        tf.math.equal(scaled - floored, 0.5), ceiled, rounded)
    quantized = tf.clip_by_value(dpu_rounded, q_min, q_max)
  return quantized


def dpu_symmetry_dequantize(inputs, scale, q_min, q_max):
  """DPU Dequantize Kernel.  DQ(x) =  x / scale. """
  with tf.name_scope("DPUDequantize"):
    return inputs / scale


def quantize_zero_point(scale, f_min, f_max, q_min, q_max):
  """Quantize the zero point. """
  with tf.name_scope("QuantizeZeroPoint"):
    f_zero_point = q_min - f_min * scale

    if f_zero_point < q_min:
      q_zero_point = q_min
    elif f_zero_point > q_max:
      q_zero_point = q_max
    else:
      q_zero_point = tf.round(f_zero_point)

    new_f_min = (q_min - q_zero_point) / scale
    new_f_max = (q_max - q_zero_point) / scale

    return q_zero_point, new_f_min, new_f_max


def get_scale(f_min, f_max, q_min, q_max):
  """Get quantize scaling factor. """
  return (q_max - q_min) / (f_max - f_min)


def get_min_max(inputs, per_channel=False):
  """Get minimum and maximum value of inputs. """
  batch_min = tf.math.reduce_min(inputs, name='batch_min')
  batch_max = tf.math.reduce_max(inputs, name='batch_max')

  range_min = tf.math.minimum(batch_min, 0.0, name='range_min')
  range_max = tf.math.maximum(batch_max, 0.0, name='range_max')
  return range_min, range_max


@tf.custom_gradient
def fake_quantize_with_quantize_pos_std(inputs, quantize_pos, bit_width):
  """The fake quantization operation kernel with std round mode.

  Args:
    inputs: a tensor containing values to be quantized.
    quantize_pos: the quantize postion
    bit_width: the bit width
  Returns:
    a tensor containing quantized values.
  """

  with tf.name_scope("FakeQuantizeWithScale"):
    bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
    bound = tf.math.pow(2.0, bit_width - 1)
    q_min = tf.math.negative(bound, name="q_min")
    q_max = tf.math.subtract(bound, 1, name="q_max")
    scale = tf.math.pow(2.0, quantize_pos, name="scale")

    quantized = symmetry_quantize(inputs, scale, q_min, q_max)
    dequantized = symmetry_dequantize(quantized, scale, q_min, q_max)

  def grad_fn(dy):
    return dy, None, None

  return dequantized, grad_fn


@tf.custom_gradient
def fake_quantize_with_quantize_pos_dpu(inputs, quantize_pos, bit_width):
  """The fake quantization operation kernel with dpu round mode.

  Args:
    inputs: a tensor containing values to be quantized.
    quantize_pos: the quantize postion
    bit_width: the bit width
  Returns:
    a tensor containing quantized values.
  """

  with tf.name_scope("FakeQuantizeWithScale"):
    bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
    bound = tf.math.pow(2.0, bit_width - 1)
    q_min = tf.math.negative(bound, name="q_min")
    q_max = tf.math.subtract(bound, 1, name="q_max")
    scale = tf.math.pow(2.0, quantize_pos, name="scale")

    quantized = dpu_symmetry_quantize(inputs, scale, q_min, q_max)
    dequantized = dpu_symmetry_dequantize(quantized, scale, q_min, q_max)

  def grad_fn(dy):
    return dy, None, None

  return dequantized, grad_fn


@tf.custom_gradient
def fake_quantize_with_log_th(inputs, log_th, bit_width):
  """The fake quantization operation kernel.

  Args:
    inputs: a tensor containing values to be quantized.
    scale: the scaling factor
    bit_width: the bit width
  Returns:
    a tensor containing quantized values.
  """

  with tf.name_scope("FakeQuantizeWithScale"):
    bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
    bound = tf.math.pow(2.0, bit_width - 1)
    q_min = tf.math.negative(bound, name="q_min")
    q_max = tf.math.subtract(bound, 1, name="q_max")
    quantize_pos = bit_width - 1 - tf.math.ceil(log_th)
    scale = tf.math.pow(2.0, quantize_pos, name="scale")

    quantized = dpu_symmetry_quantize(inputs, scale, q_min, q_max)
    dequantized = dpu_symmetry_dequantize(quantized, scale, q_min, q_max)

  def grad_fn(dy):
    # grad_wrt_inputs = 1 if f_min < x < f_max else 0
    #                         [x * s] / s - x,  if q_min < [x * s] < q_max
    # grad_wrt_log_th = ln2 * q_min / s,        if [x * s] < f_min
    #                         q_max / s,        if [x * s] > f_max
    scaled = inputs * scale
    rounded = tf.math.round(scaled)
    between_min_max = (rounded >= q_min) & (rounded <= q_max)
    ones = tf.ones_like(dy)
    zeros = tf.zeros_like(dy)
    grad_wrt_inputs = dy * tf.where(between_min_max, ones, zeros)
    grad_wrt_log_th = tf.reduce_sum(
        dy * tf.math.log(2.0) *
        tf.where(between_min_max, dequantized - inputs, quantized / scale))

    return grad_wrt_inputs, grad_wrt_log_th, None

  return dequantized, grad_fn


@tf.custom_gradient
def fake_quantize_with_min_max(inputs,
                               f_min,
                               f_max,
                               bit_width,
                               quant_zero=True):
  """The fake quantization operation kernel.

  Args:
    inputs: a tensor containing values to be quantized.
    f_min: the minimum input value
    f_max: the maximum input value
    bit_width: the bit width
  Returns:
    a tensor containing quantized values.
  """

  @tf.function
  def forward(inputs, f_min, f_max, bit_width, quant_zero):
    with tf.name_scope("FakeQuantizeWithMinMax"):
      float_bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
      bound = tf.math.pow(2.0, float_bit_width - 1)
      q_min = tf.math.negative(bound, name="q_min")
      q_max = tf.math.subtract(bound, 1, name="q_max")

      scale = get_scale(f_min, f_max, q_min, q_max)
      if quant_zero:
        q_zero_point, new_f_min, new_f_max = quantize_zero_point(
            scale, f_min, f_max, q_min, q_max)
      shift = new_f_min if quant_zero else f_min

      quantized = quantize(inputs, scale, shift, q_min, q_max)
      dequantized = dequantize(quantized, scale, shift, q_min, q_max)
      return dequantized

  @tf.function
  def grad_fn(dy):
    float_bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
    bound = tf.math.pow(2.0, float_bit_width - 1)
    q_min = tf.math.negative(bound, name="q_min")
    q_max = tf.math.subtract(bound, 1, name="q_max")

    scale = get_scale(f_min, f_max, q_min, q_max)
    if quant_zero:
      q_zero_point, new_f_min, new_f_max = quantize_zero_point(
          scale, f_min, f_max, q_min, q_max)
      between_min_max = (inputs >= new_f_min) & (inputs <= new_f_max)
      below_min = (inputs <= new_f_min)
      above_max = (inputs >= new_f_max)
    else:
      between_min_max = (inputs >= f_min) & (inputs <= f_max)
      below_min = (inputs <= f_min)
      above_max = (inputs >= f_max)

    ones = tf.ones_like(dy)
    zeros = tf.zeros_like(dy)
    grad_wrt_inputs = dy * tf.where(between_min_max, ones, zeros)
    grad_wrt_f_min = tf.reduce_sum(dy * tf.where(below_min, ones, zeros))
    grad_wrt_f_max = tf.reduce_sum(dy * tf.where(above_max, ones, zeros))
    return grad_wrt_inputs, grad_wrt_f_min, grad_wrt_f_max, None

  results = forward(inputs, f_min, f_max, bit_width, quant_zero)
  return results, grad_fn


def get_quantize_pos_non_overflow(inputs, f_min, f_max, q_min, q_max):
  """Get quantize pos which makes no value overflows. """
  with tf.name_scope("GetQuantizePosNonOverflow"):
    min_scale_inv = tf.math.divide(f_min, q_min)
    max_scale_inv = tf.math.divide(f_max, q_max)
    float_scale_inv = tf.math.maximum(min_scale_inv, max_scale_inv)

    def calc_pos():
      quantize_pos = tf.math.divide(
          tf.math.log(float_scale_inv), -tf.math.log(2.0))
      quantize_pos = tf.math.floor(quantize_pos)
      return quantize_pos

    return tf.cond(float_scale_inv < 1e-9, lambda: 127.0, calc_pos)


def get_quantize_pos_min_diffs(inputs, f_min, f_max, q_min, q_max, bit_width):
  """Get quantize pos which makes min difference between float and quantzed. """
  with tf.name_scope("GetQuantizePosMinDiffs"):
    min_scale_inv = tf.math.divide(f_min, q_min)
    max_scale_inv = tf.math.divide(f_max, q_max)
    float_scale_inv = tf.math.maximum(min_scale_inv, max_scale_inv)
    non_overflow_pos = get_quantize_pos_non_overflow(inputs, f_min, f_max,
                                                     q_min, q_max)

    def calc_pos():
      diffs = []
      for i in range(5):
        with tf.name_scope("FakeQuantizeWithScale_{}".format(i)):
          # fake quantize
          scale = tf.math.pow(2.0, non_overflow_pos + i, name="scale")
          quantized = dpu_symmetry_quantize(inputs, scale, q_min, q_max)
          dequantized = dpu_symmetry_dequantize(quantized, scale, q_min, q_max)

          diff = tf.pow(inputs - dequantized, 2)
          diff = tf.reduce_sum(diff)
          diffs.append(diff)
      pos_offset = tf.argmin(diffs)
      quantize_pos = non_overflow_pos + tf.cast(pos_offset, tf.float32)
      return quantize_pos

    return tf.cond(float_scale_inv < 1e-9, lambda: 127.0, calc_pos)


def get_quantize_pos(inputs, f_min, f_max, bit_width, method):
  """Interface function to get quantize pos. """
  bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
  bound = tf.math.pow(2.0, bit_width - 1)
  q_min = tf.math.negative(bound, name="q_min")
  q_max = tf.math.subtract(bound, 1, name="q_max")

  with tf.name_scope("GetQuantizePos"):
    if method == 0:
      return get_quantize_pos_non_overflow(inputs, f_min, f_max, q_min, q_max)
    elif method == 1:
      return get_quantize_pos_min_diffs(inputs, f_min, f_max, q_min, q_max,
                                        bit_width)
    else:
      raise NotImplementedError()


def LastValueMinMaxQuantize(inputs,
                            min_var,
                            max_var,
                            bit_width,
                            is_training,
                            mode,
                            name_scope="LastValueMinMaxQuantize"):
  """Last value float scale quantize op. """
  with tf.name_scope(name_scope):
    # ANALYSE branch
    if mode == 'ANALYSE':
      batch_min, batch_max = get_min_max(inputs)
      assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
      assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')
      return tf.identity(inputs, name='identity')

    if is_training or mode == 'QCB':
      # Training and calibration branch
      batch_min, batch_max = get_min_max(inputs)
      assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
      assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')
      return fake_quantize_with_min_max(inputs, assign_min, assign_max,
                                        bit_width)

    else:
      # Evaluation branch
      return fake_quantize_with_min_max(inputs, min_var, max_var, bit_width)


def LastValueQuantPosQuantize(inputs,
                              quant_pos_var,
                              min_var,
                              max_var,
                              bit_width,
                              method,
                              is_training,
                              mode,
                              round_mode,
                              name_scope="LastValueQuantPosQuantize"):
  """Last value power of 2 quantize op with quantize position. """
  with tf.name_scope(name_scope):
    # ANALYSE branch
    if mode == 'ANALYSE':
      batch_min, batch_max = get_min_max(inputs)
      assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
      assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')
      return tf.identity(inputs, name='identity')

    if is_training or mode == 'QCB':
      # Training and calibration branch
      batch_min, batch_max = get_min_max(inputs)
      assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
      assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')

      batch_quantize_pos = get_quantize_pos(inputs, assign_min, assign_max,
                                            bit_width, method)
      assign_quantize_pos = tf_compat.assign(
          quant_pos_var, batch_quantize_pos, name="assign_quantize_pos")

      if round_mode == 0:
        return fake_quantize_with_quantize_pos_std(inputs, assign_quantize_pos,
                                                   bit_width)
      elif round_mode == 1:
        return fake_quantize_with_quantize_pos_dpu(inputs, assign_quantize_pos,
                                                   bit_width)
      else:
        raise ValueError('Invalid round mode: {}'.format(round_mode))
    else:
      # Evaluation branch
      if round_mode == 0:
        return fake_quantize_with_quantize_pos_std(inputs, quant_pos_var,
                                                   bit_width)
      elif round_mode == 1:
        return fake_quantize_with_quantize_pos_dpu(inputs, quant_pos_var,
                                                   bit_width)
      else:
        raise ValueError('Invalid round mode: {}'.format(round_mode))


def LastValueLogThQuantize(inputs,
                           log_th_var,
                           min_var,
                           max_var,
                           bit_width,
                           is_training,
                           mode,
                           name_scope="LastValueLogThQuantize"):
  """Last value power of 2 quantize op with log threshold. """
  with tf.name_scope(name_scope):
    # ANALYSE branch
    if mode == 'ANALYSE':
      batch_min, batch_max = get_min_max(inputs)
      assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
      assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')
      return tf.identity(inputs, name='identity')

    if is_training or mode == 'QCB':
      # Training and calibration branch
      batch_min, batch_max = get_min_max(inputs)
      assign_min = tf_compat.assign(min_var, batch_min, name='assign_min')
      assign_max = tf_compat.assign(max_var, batch_max, name='assign_max')
      return fake_quantize_with_log_th(inputs, log_th_var, bit_width)

    else:
      # Evaluation branch
      return fake_quantize_with_log_th(inputs, log_th_var, bit_width)
