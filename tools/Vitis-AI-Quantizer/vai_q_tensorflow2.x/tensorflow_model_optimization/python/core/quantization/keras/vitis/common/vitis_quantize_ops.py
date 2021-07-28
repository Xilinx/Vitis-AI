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
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

logger = common_utils.VAILogger


def std_round(x):
  """ROUND_HALF_AWAY_FROM_ZERO, used in std round/py2 round.
      f(x) = std::round(x)
             ceil(x),   x - floor(x) == 0.5 && x > 0
           = round(x),  x - floor(x) != 0.5
             floor(x),  x - floor(x) == 0.5 && x < 0
      eg: f(2.3) = 2, f(1.5) = 2, f(-1.5) = -2, f(2.5) = 3, f(-2.5) = -3, f(-2.6) = -3
  """
  floored = tf.math.floor(x)
  ceiled = tf.math.ceil(x)
  rounded = tf.math.round(x)
  rounded_half = tf.where(x > 0, ceiled, floored)
  rounded = tf.where(tf.math.equal(x - floored, 0.5), rounded_half, rounded)
  return rounded


def py3_round(x):
  """ROUND_HALF_TO_EVEN, used in py3 round, tf.round or numpy.round.
      f(x) = round(x)
      eg: f(2.3) = 2, f(1.5) = 2, f(-1.5) = -2, f(2.5) = 2, f(-2.5) = -2, f(-2.6) = -3
  """
  rounded = tf.math.round(x)
  return rounded


def dpu_round(x):
  """ROUND_HALF_UP, used in dpu round.
      f(x) = (x - floor(x) == 0.5) ? ceil(x) : round(x)
           = floor(x + 0.5)
      eg: f(2.3) = 2, f(1.5) = 2, f(-1.5) = -1, f(2.5) = 3, f(-2.5) = -2, f(-2.6) = -3
  """
  rounded = tf.math.floor(x + 0.5)
  return rounded


def py3_asym_quantize(inputs, scale, shift, q_min, q_max):
  """Quantize Kernel.  Q(x) = q_min + round[(x-shift) * scale]. """
  with tf.name_scope("Py3AsymQuantize"):
    rounded = py3_round((inputs - shift) * scale)
    quantized = tf.clip_by_value(q_min + rounded, q_min, q_max)
  return quantized


def dpu_asym_quantize(inputs, scale, q_min, q_max):
  """DPU Quantize Kernel.  Q(x) = q_min + dpu_round[(x - shift) * scale]. 
  """
  with tf.name_scope("DPUAsymQuantize"):
    rounded = dpu_round((inputs - shift) * scale)
    quantized = tf.clip_by_value(q_min + rounded, q_min, q_max)
  return quantized


def py3_sym_quantize(inputs, scale, q_min, q_max):
  """Quantize Kernel.  Q(x) = round[(x) * scale]. """
  with tf.name_scope("Py3SymQuantize"):
    rounded = py3_round(inputs * scale)
    quantized = tf.clip_by_value(rounded, q_min, q_max)
  return quantized


def dpu_sym_quantize(inputs, scale, q_min, q_max):
  """DPU Quantize Kernel.  Q(x) = dpu_round[(x) * scale]. 
  """
  with tf.name_scope("DpuSymQuantize"):
    rounded = dpu_round(inputs * scale)
    quantized = tf.clip_by_value(rounded, q_min, q_max)
  return quantized


def asym_dequantize(inputs, scale, shift, q_min, q_max):
  """Dequantize Kernel.  DQ(x) =  (x - q_min) / scale + shift. """
  with tf.name_scope("AsymDequantize"):
    return (inputs - q_min) / scale + shift


def sym_dequantize(inputs, scale, q_min, q_max):
  """Dequantize Kernel.  DQ(x) =  x / scale. """
  with tf.name_scope("SymDequantize"):
    return inputs / scale


@tf.function
def quantize_zero_point(scale, f_min, f_max, q_min, q_max):
  """Quantize the zero point. """
  with tf.name_scope("QuantizeZeroPoint"):
    f_zero_point = q_min - f_min * scale

    if f_zero_point < q_min:
      q_zero_point = q_min
    elif f_zero_point > q_max:
      q_zero_point = q_max
    else:
      q_zero_point = py3_round(f_zero_point)

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
def fake_quantize_with_min_max_py3_asym(inputs, f_min, f_max, bit_width):
  """The fake quantization operation kernel with py3 asymmetry round mode.

  Args:
    inputs: a tensor containing values to be quantized.
    f_min: the minimum input value
    f_max: the maximum input value
    bit_width: the bit width
  Returns:
    a tensor containing quantized values.
  """

  with tf.name_scope("FakeQuantizeWithMinMaxPy3Asym"):
    float_bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
    bound = tf.math.pow(2.0, float_bit_width - 1)
    q_min = tf.math.negative(bound, name="q_min")
    q_max = tf.math.subtract(bound, 1, name="q_max")

    scale = get_scale(f_min, f_max, q_min, q_max)
    q_zero_point, new_f_min, new_f_max = quantize_zero_point(
        scale, f_min, f_max, q_min, q_max)
    shift = new_f_min

    quantized = py3_asym_quantize(inputs, scale, shift, q_min, q_max)
    dequantized = asym_dequantize(quantized, scale, shift, q_min, q_max)

  def grad_fn(dy):
    q_zero_point, new_f_min, new_f_max = quantize_zero_point(
        scale, f_min, f_max, q_min, q_max)
    between_min_max = (inputs >= new_f_min) & (inputs <= new_f_max)
    below_min = (inputs <= new_f_min)
    above_max = (inputs >= new_f_max)

    ones = tf.ones_like(dy)
    zeros = tf.zeros_like(dy)
    grad_wrt_inputs = dy * tf.where(between_min_max, ones, zeros)
    grad_wrt_f_min = tf.reduce_sum(dy * tf.where(below_min, ones, zeros))
    grad_wrt_f_max = tf.reduce_sum(dy * tf.where(above_max, ones, zeros))
    return grad_wrt_inputs, grad_wrt_f_min, grad_wrt_f_max, None

  return dequantized, grad_fn


@tf.custom_gradient
def fake_quantize_with_min_max_py3_sym(inputs, f_min, f_max, bit_width):
  """The fake quantization operation kernel with py3 symmetry round mode.

  Args:
    inputs: a tensor containing values to be quantized.
    f_min: the minimum input value
    f_max: the maximum input value
    bit_width: the bit width
  Returns:
    a tensor containing quantized values.
  """

  with tf.name_scope("FakeQuantizeWithMinMaxPy3Sym"):
    float_bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
    bound = tf.math.pow(2.0, float_bit_width - 1)
    q_min = tf.math.negative(bound, name="q_min")
    q_max = tf.math.subtract(bound, 1, name="q_max")

    scale = get_scale(f_min, f_max, q_min, q_max)

    quantized = py3_sym_quantize(inputs, scale, q_min, q_max)
    dequantized = sym_dequantize(quantized, scale, q_min, q_max)

  def grad_fn(dy):
    between_min_max = (inputs >= f_min) & (inputs <= f_max)
    below_min = (inputs <= f_min)
    above_max = (inputs >= f_max)

    ones = tf.ones_like(dy)
    zeros = tf.zeros_like(dy)
    grad_wrt_inputs = dy * tf.where(between_min_max, ones, zeros)
    grad_wrt_f_min = tf.reduce_sum(dy * tf.where(below_min, ones, zeros))
    grad_wrt_f_max = tf.reduce_sum(dy * tf.where(above_max, ones, zeros))
    return grad_wrt_inputs, grad_wrt_f_min, grad_wrt_f_max, None

  return dequantized, grad_fn


@tf.custom_gradient
def fake_quantize_with_quantize_pos_py3_sym(inputs, quantize_pos, bit_width):
  """The fake quantization operation kernel with py3 symmetry round mode.

  Args:
    inputs: a tensor containing values to be quantized.
    quantize_pos: the quantize postion
    bit_width: the bit width
  Returns:
    a tensor containing quantized values.
  """

  with tf.name_scope("FakeQuantizeWithQuantizePosPy3Sym"):
    bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
    bound = tf.math.pow(2.0, bit_width - 1)
    q_min = tf.math.negative(bound, name="q_min")
    q_max = tf.math.subtract(bound, 1, name="q_max")
    scale = tf.math.pow(2.0, quantize_pos, name="scale")

    quantized = py3_sym_quantize(inputs, scale, q_min, q_max)
    dequantized = sym_dequantize(quantized, scale, q_min, q_max)

  def grad_fn(dy):
    return dy, None, None

  return dequantized, grad_fn


@tf.custom_gradient
def fake_quantize_with_quantize_pos_dpu_sym(inputs, quantize_pos, bit_width):
  """The fake quantization operation kernel with dpu symmetry round mode.

  Args:
    inputs: a tensor containing values to be quantized.
    quantize_pos: the quantize postion
    bit_width: the bit width
  Returns:
    a tensor containing quantized values.
  """

  with tf.name_scope("FakeQuantizeWithQuantizePosDpuSym"):
    bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
    bound = tf.math.pow(2.0, bit_width - 1)
    q_min = tf.math.negative(bound, name="q_min")
    q_max = tf.math.subtract(bound, 1, name="q_max")
    scale = tf.math.pow(2.0, quantize_pos, name="scale")

    quantized = dpu_sym_quantize(inputs, scale, q_min, q_max)
    dequantized = sym_dequantize(quantized, scale, q_min, q_max)

  def grad_fn(dy):
    return dy, None, None

  return dequantized, grad_fn


@tf.custom_gradient
def fake_quantize_with_log_th_py3_sym(inputs, log_th, bit_width):
  """The fake quantization operation kernel with py3 symmetry round mode

  Args:
    inputs: a tensor containing values to be quantized.
    scale: the scaling factor
    bit_width: the bit width
  Returns:
    a tensor containing quantized values.
  """

  with tf.name_scope("FakeQuantizeWithLogThPy3Sym"):
    bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
    bound = tf.math.pow(2.0, bit_width - 1)
    q_min = tf.math.negative(bound, name="q_min")
    q_max = tf.math.subtract(bound, 1, name="q_max")
    quantize_pos = bit_width - 1 - tf.math.ceil(log_th)
    scale = tf.math.pow(2.0, quantize_pos, name="scale")

    quantized = py3_sym_quantize(inputs, scale, q_min, q_max)
    dequantized = sym_dequantize(quantized, scale, q_min, q_max)

  def grad_fn(dy):
    # grad_wrt_inputs = 1 if f_min < x < f_max else 0
    #                         [x * s] / s - x,  if q_min < [x * s] < q_max
    # grad_wrt_log_th = ln2 * q_min / s,        if [x * s] < f_min
    #                         q_max / s,        if [x * s] > f_max
    scaled = inputs * scale
    rounded = py3_round(scaled)
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
def fake_quantize_with_log_th_dpu_sym(inputs, log_th, bit_width):
  """The fake quantization operation kernel with dpu symmetry round mode

  Args:
    inputs: a tensor containing values to be quantized.
    scale: the scaling factor
    bit_width: the bit width
  Returns:
    a tensor containing quantized values.
  """

  with tf.name_scope("FakeQuantizeWithLogTh"):
    bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
    bound = tf.math.pow(2.0, bit_width - 1)
    q_min = tf.math.negative(bound, name="q_min")
    q_max = tf.math.subtract(bound, 1, name="q_max")
    quantize_pos = bit_width - 1 - tf.math.ceil(log_th)
    scale = tf.math.pow(2.0, quantize_pos, name="scale")

    quantized = dpu_sym_quantize(inputs, scale, q_min, q_max)
    dequantized = sym_dequantize(quantized, scale, q_min, q_max)

  def grad_fn(dy):
    # grad_wrt_inputs = 1 if f_min < x < f_max else 0
    #                         [x * s] / s - x,  if q_min < [x * s] < q_max
    # grad_wrt_log_th = ln2 * q_min / s,        if [x * s] < f_min
    #                         q_max / s,        if [x * s] > f_max
    scaled = inputs * scale
    rounded = dpu_round(scaled)
    between_min_max = (rounded >= q_min) & (rounded <= q_max)
    ones = tf.ones_like(dy)
    zeros = tf.zeros_like(dy)
    grad_wrt_inputs = dy * tf.where(between_min_max, ones, zeros)
    grad_wrt_log_th = tf.reduce_sum(
        dy * tf.math.log(2.0) *
        tf.where(between_min_max, dequantized - inputs, quantized / scale))

    return grad_wrt_inputs, grad_wrt_log_th, None

  return dequantized, grad_fn


_QUANTIZE_KERNEL_MAP = {
    'MIN_MAX_PY3_SYM': fake_quantize_with_min_max_py3_sym,
    'MIN_MAX_PY3_ASYM': fake_quantize_with_min_max_py3_asym,
    'QUANTIZE_POS_PY3_SYM': fake_quantize_with_quantize_pos_py3_sym,
    'QUANTIZE_POS_DPU_SYM': fake_quantize_with_quantize_pos_dpu_sym,
    'LOG_TH_PY3_SYM': fake_quantize_with_log_th_py3_sym,
    'LOG_TH_DPU_SYM': fake_quantize_with_log_th_dpu_sym,
}


def get_quantize_kernel(kernel_type, round_mode, symmetry=True):
  key = kernel_type
  if round_mode == 0:
    key += '_PY3'
  elif round_mode == 1:
    key += '_DPU'
  elif round_mode == 2:
    key += '_STD'
  else:
    logger.error('Invalid round mode: {}'.format(round_mode))

  if symmetry:
    key += '_SYM'
  else:
    key += '_ASYM'

  if key not in _QUANTIZE_KERNEL_MAP:
    logger.error('Invalid quantize kernel {}'.format(key))

  return _QUANTIZE_KERNEL_MAP[key]


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


def get_quantize_pos_min_diffs_py3_sym(inputs, f_min, f_max, q_min, q_max,
                                       bit_width):
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
          quantized = py3_sym_quantize(inputs, scale, q_min, q_max)
          dequantized = sym_dequantize(quantized, scale, q_min, q_max)

          diff = tf.pow(inputs - dequantized, 2)
          diff = tf.reduce_sum(diff)
          diffs.append(diff)
      pos_offset = tf.argmin(diffs)
      quantize_pos = non_overflow_pos + tf.cast(pos_offset, tf.float32)
      return quantize_pos

    return tf.cond(float_scale_inv < 1e-9, lambda: 127.0, calc_pos)


def get_quantize_pos_min_diffs_dpu_sym(inputs, f_min, f_max, q_min, q_max,
                                       bit_width):
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
          quantized = dpu_sym_quantize(inputs, scale, q_min, q_max)
          dequantized = sym_dequantize(quantized, scale, q_min, q_max)

          diff = tf.pow(inputs - dequantized, 2)
          diff = tf.reduce_sum(diff)
          diffs.append(diff)
      pos_offset = tf.argmin(diffs)
      quantize_pos = non_overflow_pos + tf.cast(pos_offset, tf.float32)
      return quantize_pos

    return tf.cond(float_scale_inv < 1e-9, lambda: 127.0, calc_pos)


def get_quantize_pos(inputs, f_min, f_max, bit_width, method, round_mode):
  """Interface function to get quantize pos. """
  bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
  bound = tf.math.pow(2.0, bit_width - 1)
  q_min = tf.math.negative(bound, name="q_min")
  q_max = tf.math.subtract(bound, 1, name="q_max")

  with tf.name_scope("GetQuantizePos"):
    if method == 0:
      return get_quantize_pos_non_overflow(inputs, f_min, f_max, q_min, q_max)
    elif method == 1 and round_mode == 0:
      return get_quantize_pos_min_diffs_py3_sym(inputs, f_min, f_max, q_min,
                                                q_max, bit_width)
    elif method == 1 and round_mode == 1:
      return get_quantize_pos_min_diffs_dpu_sym(inputs, f_min, f_max, q_min,
                                                q_max, bit_width)
    else:
      logger.error('', NotImplementedError)


def get_log_th_non_overflow(inputs, f_min, f_max, q_min, q_max):
  """Get log threshold which makes no value overflows. """
  with tf.name_scope("GetLogThNonOverflow"):
    f_min_abs = tf.math.abs(f_min)
    f_max_adj = f_max * tf.math.divide(-q_min, q_max)
    th = tf.math.maximum(f_min_abs, f_max_adj)
    th = tf.math.maximum(th, 1e-9)
    return tf.math.divide(tf.math.log(th), tf.math.log(2.))


def get_log_th(inputs, f_min, f_max, bit_width, method):
  """Interface function to get log threshold. """
  bit_width = tf.cast(bit_width, dtype=tf.float32, name="bit_width")
  bound = tf.math.pow(2.0, bit_width - 1)
  q_min = tf.math.negative(bound, name="q_min")
  q_max = tf.math.subtract(bound, 1, name="q_max")

  with tf.name_scope("GetLogTh"):
    if method == 0:
      return get_log_th_non_overflow(inputs, f_min, f_max, q_min, q_max)
    elif method == 1:
      logger.error('Method 1 not implemented.', NotImplementedError)
    else:
      logger.error('Method {} not implemented.'.format(method),
                   NotImplementedError)


def LastValueMinMaxQuantize(inputs,
                            min_var,
                            max_var,
                            bit_width,
                            round_mode,
                            mode,
                            is_training,
                            name_scope="LastValueMinMaxQuantize"):
  """Last value float scale quantize op.

  Args:
    inputs: Input values.
    min_var: Variable of minimum value of inputs.
    max_var: Variable of maximum value of inputs.
    bit_width: Int, bit width of quantized values.
    round_mode: Int, the mode of rounding function, 0 for PY3 round. Now only PY3 round is supported.
    mode: String, the mode of quantization, available modes are ['ANALYSE', 'QCB', 'QCBEV', 'QAT']
    is_training: Bool, whether in training phase.

  Return:
    Quantized inputs.
  """
  with tf.name_scope(name_scope):

    quantize_kernel = get_quantize_kernel(
        kernel_type='MIN_MAX', round_mode=round_mode)

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
      return quantize_kernel(inputs, assign_min, assign_max, bit_width)
    else:
      # Evaluation branch
      return quantize_kernel(inputs, min_var, max_var, bit_width)


def LastValueQuantPosQuantize(inputs,
                              quant_pos_var,
                              min_var,
                              max_var,
                              bit_width,
                              method,
                              round_mode,
                              mode,
                              is_training,
                              name_scope="LastValueQuantPosQuantize"):
  """Last value power of 2 quantize op with quantize position. 

  Args:
    inputs: Input values.
    quant_pos_var: Variable of quantize position.
    min_var: Variable of minimum value of inputs.
    max_var: Variable of maximum value of inputs.
    bit_width: Int, bit width of quantized values.
    method: Int, method of how to get the quantize pos, 0 for non_overflow and 1 for min_diffs.
    round_mode: Int, the mode of rounding function, 0 for PY3 round, 1 for DPU round.
      By default, weights are quantized with PY3 round, inputs and activations are quantized with DPU round.
    mode: String, the mode of quantization, available modes are ['ANALYSE', 'QCB', 'QCBEV', 'QAT']
    is_training: Bool, whether in training phase.

  Return:
    Quantized inputs.
  """
  with tf.name_scope(name_scope):

    quantize_kernel = get_quantize_kernel(
        kernel_type='QUANTIZE_POS', round_mode=round_mode)

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
                                            bit_width, method, round_mode)
      assign_quantize_pos = tf_compat.assign(
          quant_pos_var, batch_quantize_pos, name="assign_quantize_pos")

      return quantize_kernel(inputs, assign_quantize_pos, bit_width)
    else:
      # Evaluation branch
      return quantize_kernel(inputs, quant_pos_var, bit_width)


def LastValueLogThQuantize(inputs,
                           log_th_var,
                           min_var,
                           max_var,
                           bit_width,
                           method,
                           round_mode,
                           mode,
                           is_training,
                           name_scope="LastValueLogThQuantize"):
  """Last value power of 2 quantize op with log threshold.

  Args:
    inputs: Input values.
    log_th_var: Variable of log threshold.
    min_var: Variable of minimum value of inputs.
    max_var: Variable of maximum value of inputs.
    bit_width: Int, bit width of quantized values.
    method: Int, method of how to get the initial log threshold, 0 for non_overflow and 1 for min_diffs.
    round_mode: Int, the mode of rounding function, 0 for PY3 round, 1 for DPU round, 2 for STD round.
    mode: String, the mode of quantization, available modes are ['ANALYSE', 'QCB', 'QCBEV', 'QAT']
    is_training: Bool, whether in training phase.

  Return:
    Quantized inputs.
  """
  with tf.name_scope(name_scope):

    quantize_kernel = get_quantize_kernel(
        kernel_type='LOG_TH', round_mode=round_mode)

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

      if mode == 'QCB':
        batch_log_th = get_log_th(inputs, assign_min, assign_max, bit_width,
                                  method)
        assign_log_th = tf_compat.assign(
            log_th_var, batch_log_th, name="assign_log_th")
        return quantize_kernel(inputs, assign_log_th, bit_width)

      else:
        return quantize_kernel(inputs, log_th_var, bit_width)

    else:
      # Evaluation branch
      return quantize_kernel(inputs, log_th_var, bit_width)
