from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import enum
import functools
import tensorflow as tf
import numpy as np

class RoundMode(enum.Enum):
  """Enum class for round modes."""
  # ROUND_HALF_TO_EVEN, used in py3 round, tf.round or numpy.round.
  HALF_TO_EVEN = 0

  # ROUND_HALF_UP, used in dpu round and tf.fake_quant.
  HALF_UP = 1

  # ROUND_HALF_AWAY_FROM_ZERO, used in std round/py2 round.
  HALF_AWAY_FROM_ZERO = 2


def round_half_to_even(x):
  """ROUND_HALF_TO_EVEN, used in py3 round, tf.round or numpy.round.
      f(x) = round(x)
      eg: f(2.3) = 2, f(1.5) = 2, f(-1.5) = -2, f(2.5) = 2, f(-2.5) = -2, f(-2.6) = -3
  """
  rounded = tf.math.round(x)
  return rounded


def round_half_up(x):
  """ROUND_HALF_UP, used in dpu round and tf.fake_quant
      f(x) = (x - floor(x) == 0.5) ? ceil(x) : round(x)
           = floor(x + 0.5)
      eg: f(2.3) = 2, f(1.5) = 2, f(-1.5) = -1, f(2.5) = 3, f(-2.5) = -2, f(-2.6) = -3
  """
  rounded = tf.math.floor(x + 0.5)
  return rounded


def round_half_away_from_zero(x):
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


def round(x, round_mode):
  """Round with different modes."""
  if round_mode == RoundMode.HALF_TO_EVEN:
    return round_half_to_even(x)
  elif round_mode == RoundMode.HALF_UP:
    return round_half_up(x)
  elif round_mode == RoundMode.HALF_AWAY_FROM_ZERO:
    return round_half_away_from_zero(x)
  else:
    logger.error('Invalid round_mode: {}'.format(round_mode))

bit_width=16
round_mode=0
round_mode = RoundMode(round_mode)
axis=-1
block_size = 16
tile_size=16
epsilon=tf.math.pow(tf.constant(2.0), -23)

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

def bfp_quantize(inputs, scale, round_mode, min_v, max_v):
  rounded = round(inputs/scale, round_mode)*scale
  quantized = tf.clip_by_value(rounded, min_v, max_v)
  return quantized

def bfp_dequantize(inputs, shape, axis):
  def get_shape(s):
    # convert (None, W, H, C) to [-1, W, H, C]
    new_shape = []
    #for i in range(s.ndims):
    for i in range(len(s)):
      if s[i]:
        new_shape.append(s[i])
      else:
        new_shape.append(-1)
    return new_shape

  #origin_n_dim = shape.ndims
  origin_n_dim = len(shape)
  _, L, B = inputs.shape
  # [N*W*H, L, B] -> [N*H*W, L*B] -> [N*W*H, C]
  inputs = tf.reshape(inputs, [-1, L*B])
  inputs = tf.slice(inputs, [0,0], [-1, shape[axis]])
  new_shape = get_shape(shape)
  if axis == -1 or axis == origin_n_dim-1: 
    dequantized = tf.reshape(inputs, new_shape)
  else:
    t = new_shape[-1]
    new_shape[-1] = new_shape[axis]
    new_shape[axis] = t
    dequantized = tf.reshape(inputs, new_shape)
    perm = get_perm(origin_n_dim, axis)
    dequantized = tf.transpose(dequantized, perm=perm)
  return dequantized

def _transform_to_block_wise(inputs):
  C = inputs.shape[axis]
  #n_dim = inputs.shape.ndims

  n_dim = len(inputs.shape)
  assert np.abs(axis) < n_dim 
  if axis != -1 and axis != n_dim-1:
    perm = get_perm(n_dim, axis)
    inputs = tf.transpose(inputs, perm=perm)
  # [N, W, H, C] -> [N*W*H, C]
  inputs = tf.reshape(inputs, [-1, C])
  padded_channels = tile_size - C % tile_size
  if padded_channels != tile_size:
    inputs = tf.pad(inputs, tf.constant([[0,0,], [0,padded_channels]]), "CONSTANT")
  # [N*W*H, C] -> [N*W*H, L, B]
  _, C = inputs.shape
  return tf.reshape(inputs, [-1, int(C/tile_size), tile_size])

def _get_exponent(inputs):
  t = tf.abs(inputs)
  # use fp32's 1.mantissa_bits
  max_t = tf.math.reduce_max(t, axis=-1, keepdims=True)
  max_exp = tf.math.floor(tf.math.log(max_t + epsilon) / tf.math.log(2.0))
  t_exp = tf.math.floor(tf.math.log(t + epsilon) / tf.math.log(2.0))
  return max_exp, t_exp

def _get_smallest_and_largest(exp):
  # sign bits: 1, exponent bits: 8, no implicit leading 1
  mantissa_bits = bit_width - 9
  # The min/max representable value with exp
  smallest = tf.math.pow(2.0, exp - (mantissa_bits - 1))
  largest = tf.math.pow(2.0, exp + 1) - smallest
  return smallest, largest

def custom_bfp_fake_quant(inputs):
  input_shape = inputs.get_shape()
  inputs = _transform_to_block_wise(inputs)
  max_exp, _ = _get_exponent(inputs)
  interval, max_v = _get_smallest_and_largest(max_exp)
  quantized = bfp_quantize(inputs, interval, round_mode, -max_v, max_v)
  dequantized = bfp_dequantize(quantized, input_shape, axis)
  return dequantized

def _get_shared_exponent(sub_max_exp):
  # sub_max_exp:  maximum exponent in one sub-block
  n = int(block_size / tile_size)
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

def _get_exponent_with_shift(shared_exp, sub_exp):
  # d: number of bits in the MSFP' sub-block shift field
  d = 1
  threshold = tf.cast(tf.pow(2, d) - 1, tf.float32)
  less = tf.less(tf.subtract(shared_exp, sub_exp), threshold)
  max_exp = tf.where(less, sub_exp, tf.subtract(shared_exp, threshold))
  return max_exp

def custom_msfp_fake_quant(inputs):
  """The custom dataformat-bfp fake quantization operation kernel.

  Args:
    inputs: a tensor containing values to be quantized.
  Returns:
    a tensor containing quantized values.
  """
  input_shape = tf.shape(inputs)
  inputs = _transform_to_block_wise(inputs)
  sub_max_exp, _ = _get_exponent(inputs)
  shared_max_exp = _get_shared_exponent(sub_max_exp)
  max_exp = _get_exponent_with_shift(shared_max_exp, sub_max_exp)
  print(max_exp)

  interval, max_v = _get_smallest_and_largest(max_exp)
  quantized = bfp_quantize(inputs, interval, round_mode, -max_v, max_v)
  dequantized = bfp_dequantize(quantized, input_shape, axis)

  return dequantized


#arr = np.random.randn(2,2,3,4)
l = [[[[ 1.71982214, 1.4036343, -1.83156142, 0.7838483 ],
   [ 0.48769499, 1.61412816, -0.12445149, 1.16040178],
   [-1.19621066, -0.93904581, 1.30315017, 1.47174008]],

  [[-0.30113716, -0.02443515, -0.55273104, 0.74774681],
   [-0.51307953, -1.27320177, 0.32037816, 1.06640902],
   [ 0.12659431, -1.46967731, -0.29188104, 0.87889924]]],


 [[[ 0.93228734, 0.56006482, -1.61691514, -0.06036616],
   [ 1.12935093, 0.82072662, 0.11518568, 0.84932466],
   [ 0.14456907, 0.30759453, -1.66119122, -0.07298027]],

  [[ 0.47681045, 0.94810189, 0.56308247, -0.59616665],
   [-0.11822137, 0.73009973, -1.76281126, -1.57564246],
   [ 0.79164054, 0.01013313, 0.37905317, -1.18901355]]]]
arr = np.array(l, dtype=np.float32)
#arr = np.random.randn(1,1,1,14)
inputs=tf.convert_to_tensor(arr, dtype=tf.float32)
print(inputs)
#inputs = tf.random.uniform([1,1,1,14], minval=0, maxval=12, dtype=tf.float32)
outputs = custom_bfp_fake_quant(inputs)
print('After bfp:\n', outputs)

#outputs = custom_msfp_fake_quant(inputs)
#print('After msfp:\n', outputs)
