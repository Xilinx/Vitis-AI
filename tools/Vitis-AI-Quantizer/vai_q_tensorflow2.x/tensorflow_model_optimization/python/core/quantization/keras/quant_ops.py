# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# TODO(b/139939526): move to public API.
from tensorflow.python.training import moving_averages
from tensorflow_model_optimization.python.core.keras import compat as tf_compat


def FixedQuantize(inputs, init_min=-6.0, init_max=6.0, scope=None):
  """Adds a fake quantize layer with fixed quantization interval.

  Args:
    inputs: a tensor containing values to be quantized.
    init_min: the lower end of quantization interval.
    init_max: the upper end of quantization interval.
    scope: Optional scope for name_scope.
  Returns:
    a tensor containing quantized values.
  """
  if scope is None:
    scope = 'FixedQuantize'

  with tf.name_scope(scope):
    return tf.quantization.fake_quant_with_min_max_args(
        inputs, min=init_min, max=init_max)


def AllValuesQuantize(inputs,
                      min_var,
                      max_var,
                      name_prefix='AllValuesQuantize',
                      is_training=True,
                      num_bits=8,
                      narrow_range=False,
                      symmetric=False):
  """Adds a layer that collects quantization ranges as min/max of tensor values.

  AllValuesQuantize creates variables called 'min' and 'max',
  representing the interval used for quantization and clamping.

  Args:
    inputs: a tensor containing values to be quantized.
    min_var: Variable which stores the min value of tensor.
    max_var: Variable which stores the max value of tensor.
    name_prefix: name_prefix for created nodes.
    is_training: Whether the op is applied to a training or eval graph.
    num_bits: Number of bits to use for quantization, must be between 2 and 8.
    narrow_range: Whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
    symmetric: If true, use symmetric quantization limits instead of training
      the minimum and maximum of each quantization range separately.
  Returns:
    a tensor containing quantized values.
  """
  with tf.name_scope(name_prefix):
    if not is_training:
      return _FakeQuantWithMinMaxVars(
          inputs,
          min_var,
          max_var,
          per_channel=False,
          num_bits=num_bits,
          narrow_range=narrow_range)

    batch_min = tf.math.reduce_min(inputs, name='BatchMin')
    batch_max = tf.math.reduce_max(inputs, name='BatchMax')

    if symmetric:
      if narrow_range:
        min_max_ratio = -1
      else:
        # In two's complement notation, the negative range is slightly larger
        # than the positive range.
        min_max_ratio = -((1 << num_bits) - 2) / (1 << num_bits)

      # TFLite requires that 0.0 is always in the [min; max] range. Because
      # batch_min <= batch_max, it follows that range_min <= 0 <= range_max.
      batch_min = tf.math.minimum(batch_min, batch_max / min_max_ratio)
      batch_max = tf.math.maximum(batch_max, batch_min * min_max_ratio)

    # TFLite requires that 0.0 if always in the [min; max] range.
    range_min = tf.math.minimum(tf.math.minimum(min_var, batch_min), 0.0)
    range_max = tf.math.maximum(tf.math.maximum(max_var, batch_max), 0.0)

    assign_min = tf_compat.assign(min_var, range_min, name='AssignMinAllValue')
    assign_max = tf_compat.assign(max_var, range_max, name='AssignMaxAllValue')

    return _FakeQuantWithMinMaxVars(
        inputs,
        assign_min,
        assign_max,
        per_channel=False,
        num_bits=num_bits,
        narrow_range=narrow_range)


def LastValueQuantize(inputs,
                      min_var,
                      max_var,
                      per_channel=False,
                      name_prefix='LastValueQuant',
                      is_training=True,
                      num_bits=8,
                      narrow_range=False,
                      symmetric=False):
  """Adds a layer that collects quantization ranges as last input ranges.

  LastValueQuantize creates variables called 'min' and 'max', representing the
  interval used for quantization and clamping.

  Args:
    inputs: a tensor containing values to be quantized.
    per_channel: (Optional) a boolean specifying whether to use different
      quantization ranges per output channel.
    init_min: a float scalar, the initial value for variable min.
    init_max: a float scalar, the initial value for variable max.
    name_prefix: name_prefix for created nodes.
    is_training: Whether the op is applied to a training or eval graph.
    num_bits: Number of bits to use for quantization, must be between 2 and 8.
    narrow_range: Whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
    symmetric: If true, use symmetric quantization limits instead of training
      the minimum and maximum of each quantization range separately.
  Returns:
    a tensor containing quantized values.
  """
  with tf.name_scope(name_prefix):
    input_shape = inputs.get_shape()
    input_dim = len(input_shape)

    if not is_training:
      return _FakeQuantWithMinMaxVars(
          inputs,
          min_var,
          max_var,
          per_channel=per_channel,
          num_bits=num_bits,
          narrow_range=narrow_range)

    if per_channel:
      if input_dim == 2:
        reduce_dims = [0]
      elif input_dim == 4:
        reduce_dims = [0, 1, 2]

    if per_channel:
      if input_dim >= 2:
        batch_min = tf.math.reduce_min(
            inputs, axis=reduce_dims, name='BatchMin')
      else:
        batch_min = inputs
    else:
      batch_min = tf.math.reduce_min(inputs, name='BatchMin')

    if per_channel:
      if input_dim >= 2:
        batch_max = tf.math.reduce_max(
            inputs, axis=reduce_dims, name='BatchMax')
      else:
        batch_max = inputs
    else:
      batch_max = tf.math.reduce_max(inputs, name='BatchMax')

    if symmetric:
      if narrow_range:
        min_max_ratio = -1
      else:
        # In two's complement notation, the negative range is slightly larger
        # than the positive range.
        min_max_ratio = -((1 << num_bits) - 2) / (1 << num_bits)

      # TFLite requires that 0.0 if always in the [min; max] range. Because
      # batch_min <= batch_max, it follows that range_min <= 0 <= range_max.
      range_min = tf.math.minimum(batch_min, batch_max / min_max_ratio)
      range_max = tf.math.maximum(batch_max, batch_min * min_max_ratio)
    else:
      # TFLite requires that 0.0 if always in the [min; max] range.
      range_min = tf.math.minimum(batch_min, 0.0)
      range_max = tf.math.maximum(batch_max, 0.0)

    assign_min = tf_compat.assign(min_var, range_min, name='AssignMinLast')
    assign_max = tf_compat.assign(max_var, range_max, name='AssignMaxLast')

    return _FakeQuantWithMinMaxVars(
        inputs,
        assign_min,
        assign_max,
        per_channel=per_channel,
        num_bits=num_bits,
        narrow_range=narrow_range)


def MovingAvgQuantize(inputs,
                      min_var,
                      max_var,
                      per_channel=False,
                      ema_decay=0.999,
                      name_prefix='MovingAvgQuantize',
                      is_training=True,
                      num_bits=8,
                      narrow_range=False,
                      symmetric=False):
  """Adds a layer that collects quantization ranges as EMAs of input ranges.

  MovingAvgQuantize creates variables called 'min' and 'max', representing the
  interval used for quantization and clamping.

  Args:
    inputs: a tensor containing values to be quantized.
    per_channel: (default False) a boolean specifying whether to use different
      quantization ranges per output channel.
    init_min: a float scalar, the initial value for variable min.
    init_max: a float scalar, the initial value for variable max.
    ema_decay: EMA decay parameter.
    name_prefix: name_prefix for created nodes.
    is_training: Whether the op is applied to a training or eval graph.
    num_bits: Number of bits to use for quantization, must be between 2 and 8.
    narrow_range: Whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
    symmetric: If true, use symmetric quantization limits instead of training
      the minimum and maximum of each quantization range separately.
  Returns:
    a tensor containing quantized values.
  """
  with tf.name_scope(name_prefix):
    input_shape = inputs.get_shape()
    input_dim = len(input_shape)

    if not is_training:
      return _FakeQuantWithMinMaxVars(
          inputs,
          min_var,
          max_var,
          per_channel=per_channel,
          num_bits=num_bits,
          narrow_range=narrow_range)
    if per_channel:
      if input_dim == 2:
        reduce_dims = [0]
      elif input_dim == 4:
        reduce_dims = [0, 1, 2]

    if per_channel:
      if input_dim >= 2:
        batch_min = tf.math.reduce_min(
            inputs, axis=reduce_dims, name='BatchMin')
      else:
        batch_min = inputs
    else:
      batch_min = tf.math.reduce_min(inputs, name='BatchMin')

    if per_channel:
      if input_dim >= 2:
        batch_max = tf.math.reduce_max(
            inputs, axis=reduce_dims, name='BatchMax')
      else:
        batch_max = inputs
    else:
      batch_max = tf.math.reduce_max(inputs, name='BatchMax')

    if symmetric:
      if narrow_range:
        min_max_ratio = -1
      else:
        # In two's complement notation, the negative range is slightly larger
        # than the positive range.
        min_max_ratio = -((1 << num_bits) - 2) / (1 << num_bits)

      # TFLite requires that 0.0 if always in the [min; max] range. Because
      # batch_min <= batch_max, it follows that range_min <= 0 <= range_max.
      range_min = tf.minimum(batch_min, batch_max / min_max_ratio)
      range_max = tf.maximum(batch_max, batch_min * min_max_ratio)
    else:
      # TFLite requires that 0.0 if always in the [min; max] range.
      range_min = tf.minimum(batch_min, 0.0)
      range_max = tf.maximum(batch_max, 0.0)

    assign_min = moving_averages.assign_moving_average(
        min_var, range_min, ema_decay, zero_debias=False, name='AssignMinEma')
    assign_max = moving_averages.assign_moving_average(
        max_var, range_max, ema_decay, zero_debias=False, name='AssignMaxEma')

    return _FakeQuantWithMinMaxVars(
        inputs,
        assign_min,
        assign_max,
        per_channel=per_channel,
        num_bits=num_bits,
        narrow_range=narrow_range)


def _FakeQuantWithMinMaxVars(inputs, min_var, max_var, per_channel, num_bits,
                             narrow_range):
  """Adds a fake quantization operation.

  Depending on value of per_channel, this operation may do global quantization
  or per channel quantization.  min_var and max_var should have corresponding
  shapes: [1] when per_channel == False and [d] when per_channel == True.

  Args:
    inputs: a tensor containing values to be quantized.
    min_var: a variable containing quantization range lower end(s).
    max_var: a variable containing quantization range upper end(s).
    per_channel: a boolean specifying whether to use per-channel quantization.
    num_bits: Number of bits to use for quantization, must be between 2 and 8.
    narrow_range: Whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
  Returns:
    a tensor containing quantized values.
  """

  if per_channel:
    assert len(min_var.get_shape()) == 1
    assert len(max_var.get_shape()) == 1
    return tf.quantization.fake_quant_with_min_max_vars_per_channel(
        inputs, min_var, max_var, num_bits=num_bits, narrow_range=narrow_range)
  else:
    assert min_var.get_shape() == []  # pylint: disable=g-explicit-bool-comparison
    assert max_var.get_shape() == []  # pylint: disable=g-explicit-bool-comparison
    return tf.quantization.fake_quant_with_min_max_vars(
        inputs, min_var, max_var, num_bits=num_bits, narrow_range=narrow_range)
