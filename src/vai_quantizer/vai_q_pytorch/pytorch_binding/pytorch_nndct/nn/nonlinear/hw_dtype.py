# Author: Ephrem (ephremw@xilinx.com)
# Gitenterprise: https://gitenterprise.xilinx.com/ephremw/nonlinear/blob/master/python/hw_dtype.py
"""Data type conversion routines """
from typing import NamedTuple
from typing import Iterator, Iterable
import math
import numpy as np
#import pyfma

def is_subnormal(x):
  """Returns a boolean `numpy` array for whether the elements in `x` are
  subnormal.
  Args:
    x (numpy array): input
  Returns:
    Boolean array indicating whether elements are subnormal
  """
  dtype = x.dtype
  min_normal = np.finfo(dtype).tiny
  return np.logical_and(
      np.logical_and(np.isfinite(x), x != 0),
      np.abs(x) < min_normal)

def is_normal(x):
  """Returns a boolean `numpy` array for whether the elements in `x` are normal.
  Args:
    x (numpy array): input
  Returns:
    Boolean array indicating whether elements are normal
  """
  dtype = x.dtype
  min_normal = np.finfo(dtype).tiny
  return np.logical_and(
      np.isfinite(x), np.logical_or(x == 0,
                                    np.abs(x) >= min_normal))

def _rsqrt_fp32(x, magics, two_iterations=False, fused=False):
  """Computes fast inverse square root with different magic numbers."""

  def newton_raphson(last_result, threehalves, half_input, mult_add_func):
    # y = y * (1.5 - 0.5 x y y)
    if mult_add_func is None:
      return (np.float32(last_result * np.float32(
          np.float32(threehalves) -
          np.float32(np.float32(half_input * last_result) * last_result))))
    tmp = np.float32(mult_add_func(half_input, last_result, 0.0))  # 0.5 x y
    tmp = np.float32(mult_add_func(-tmp, last_result,
                                   threehalves))  # -0.5xy^2+1.5
    return np.float32(mult_add_func(last_result, tmp, 0.0))

  mult_add_func = lambda x, y, z: mult_add(
      x=x,
      y=y,
      z=z,
      x_dtype=fp32,
      y_dtype=fp32,
      z_dtype=fp32,
      out_dtype=fp32,
      fused=fused)

  threehalves_1, threehalves_2 = magics[1:3]
  half_x = np.float32(x / 2)
  x_as_int = np.array(x, dtype=np.float32).view(dtype=np.int32)
  x_as_int = (magics[0] - np.int32(x_as_int >> 1)).astype(np.int32)
  result = x_as_int.view(dtype=np.float32)
  result = newton_raphson(result, threehalves_1, half_x, mult_add_func)
  if two_iterations:
    result = newton_raphson(result, threehalves_2, half_x, mult_add_func)
  #result = np.float32(
  #    result * np.float32(
  #        np.float32(threehalves_1) - np.float32(np.float32(half_x * result)*result)))
  #if two_iterations:
  #  result = np.float32(
  #      result * np.float32(
  #          np.float32(threehalves_2) - np.float32(np.float32(half_x * result)*result)))
  return result

def rsqrt_fp32(x, two_iterations=False, fused=False):
  """Fast inverse square root used in Quake. See :py:func:`rsqrt_fp32_walczyk`
  for an alternative version."""
  return _rsqrt_fp32(
      x,
      two_iterations=two_iterations,
      magics=[0x5f3759df, np.float32(1.5),
              np.float32(1.5)],
      fused=fused)

def rsqrt_fp32_walczyk(x, two_iterations=False, fused=False):
  """ See Walczyk,C.J.;Moroz,L.V.;Cies ́lin ́ski,J.L.,
  Improving the Accuracy of the Fast Inverse Square Root by Modifying
  Newton–Raphson Corrections. Entropy 2021, 23, 86.
  """
  return _rsqrt_fp32(
      x,
      two_iterations=two_iterations,
      magics=[0x5f376908,
              np.float32(1.50087896),
              np.float32(1.50000057)],
      fused=fused)

def vmath_ulp(x):
  """Vectorized version of `math.ulp(x)`"""
  return np.vectorize(math.ulp)(x)

def vhex(x):
  """Vectorized hex conversion"""

  def _hex(x):
    return x.hex()

  return np.vectorize(_hex)(x)

def goldberg_ulp_err(x, ref, dtype_str):
  """David Goldberg's definition of errors in ulp"""
  x = to_dtype(x, dtype_str)  # uses precision of dtype_str
  fraction, exponent = np.frexp(x)
  normalized_sig, normalized_exp = fraction * 2, exponent - 1
  return (np.abs(normalized_sig - ref * np.exp2(-normalized_exp)) *
          np.exp2(PRECISIONS[dtype_str] - 1))

class Dtype(NamedTuple):
  """Numerical data type attributes
  Attributes:
    name (str): Name of the data type
    round (str): `rn` for round-to-nearest and `rd` for truncation (round-down)
  """
  name: str
  round: str
  emin: int
  emax: int
  precision: int

  def __repr__(self):
    return self.name

  def string_to_dtype(self):
    """Returns :py:type:`Dtype` from string name"""
    return DTYPES[self.name]

fp16 = Dtype('fp16', 'rn', -30, 15, 11)
"""FP16 shorthand"""
bf16 = Dtype('bf16', 'rn', -126, 127, 8)
"""Google Brain Float16 shorthand"""
tf32 = Dtype('tf32', 'rn', -126, 127, 11)
"""Tensor Float16 shorthand"""
fp32 = Dtype('fp32', 'rn', -126, 127, 24)
"""FP32 shorthand"""
fp64 = Dtype('fp64', 'rn', -1022, 1023, 53)
"""FP64 shorthand"""

# Aliases
PRECISIONS, EXPONENT_RANGES, DTYPES = {}, {}, {}
for k in [fp16, bf16, tf32, fp32, fp64]:
  PRECISIONS[k.name] = k.precision
  EXPONENT_RANGES[k.name] = range(k.emin, k.emax + 1)
  DTYPES[k.name] = k

def min_positive_subnormal(dtype_str: str):
  """Returns the smallest positive subnormal number"""
  return np.exp2(EXPONENT_RANGES[dtype_str][0] - (PRECISIONS[dtype_str] - 1))

def to_dtype(x, dtype_str, mode='rn'):
  """Converts an input array to another data type.
  Args:
    x (ArrayLike): Input array
    dtype_str: Use `bf16` or `bfloat16` for the Google Brain 16-bit
      floating-point format, `tf32` for the Nvidia TF32 format, which has the
      precsion of fp16 and the dynamic range of fp32, `fp16` or `float16` for
      IEEE `binary16`, and `fp32` or `float32` for IEEE `binary32`.
    mode (str): Use `rn` for round-to-nearest and `rd` for round-down
      (truncation)
  Returns:
    Array convernd to the requested data type
  """
  if dtype_str in ['bf16', 'bfloat16']:
    return to_bf16(x, mode=mode)
  if dtype_str == 'tf32':
    return to_tf32(x, mode=mode)
  if dtype_str in ['fp16', 'float16']:
    assert mode == 'rn'
    return to_fp16(x)
  if dtype_str in ['fp32', 'float32']:
    assert mode == 'rn'
    return to_fp32(x)
  if dtype_str in ['fp64', 'float64']:
    assert mode == 'rn'
    return to_fp64(x)
  return x

def to_fp32(x):
  """Returns `np.float32(x)`"""
  return np.float32(x)

def to_fp16(x):
  """Returns `np.float16(x)`"""
  return np.float16(x)

def to_fp64(x):
  """Returns `np.float64(x)`"""
  return np.float64(x)

def to_bf16(x, mode='rn'):
  """Returns Google Brain Float16 (bfloat16) format by truncation.
  The `bfloat16` format is a 16-bit field with a 1-bit sign, an 8-bit exponent,
  and a 7-bit trailing significand field (a.k.a. mantissa in older
  specifications). Google's default casting routine converts an IEEE `binary32`
  (float32) to bfloat16 by copying the upper 16 bits.
  Since `numpy` does not natively support `bfloat16`, this routine returns
  a `bfloat16` number as a `numpy.float32` rounded according to `mode`.
  .. code-block:: c
    static inline tensorflow::bfloat16 FloatToBFloat16(float float_val) {
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        return *reinterpret_cast<tensorflow::bfloat16*>(
            reinterpret_cast<uint16_t*>(&float_val));
    #else
        return *reinterpret_cast<tensorflow::bfloat16*>(
            &(reinterpret_cast<uint16_t*>(&float_val)[1]));
    #endif
    }
  Args:
    x (ArrayLike): Input array
    mode (str): Use `rn` for round-to-nearest and `rd` for round-down
      (truncation)
  Returns:
    Bfloat16 data
  """
  return np.float32(round_msbs(x, PRECISIONS['bf16'], mode))

def to_tf32(x, mode='rn'):
  """Converts an input array to the TF32 format (1, 8, 10).
  Args:
    x (ArrayLike): Input array
    mode (str): Use `rn` for round-to-nearest and `rd` for round-down
      (truncation)
  Returns:
    TF32 data
  """
  return np.float32(round_msbs(x, PRECISIONS['tf32'], mode))

def round_msbs(x, sig_bits, mode='rn'):
  """Rounds `x` to `n` signficant bits. Supports truncation and
  round-to-nearest-even.
  Args:
    x (ArrayLike): Input array to be rounded
    sig_bits: (int) Number of significant bits (normalization bit plus trailing signficand).
      For the IEEE binary32 format, this value is 24.
    mode (str): Use `rn` for round-to-nearest and `rd` for round-down
      (truncation)
  Returns:
    Rounded array
  """
  fracs, exps = np.frexp(x)  # fracs in open interval (-1, 1)
  fracs = (np.trunc(fracs * np.power(2, sig_bits)) if mode == 'rd' else np.rint(
      fracs * np.power(2, sig_bits))) / np.power(2, sig_bits)
  abs_fracs = np.abs(fracs)
  fracs = np.where(abs_fracs >= 1, fracs / 2, fracs)
  exps = np.where(abs_fracs >= 1, exps + 1, exps)
  return np.ldexp(fracs, exps)

def mult(x, y, x_dtype: Dtype, y_dtype: Dtype, out_dtype: Dtype = None):
  """Multiplies two numbers of one type and outputs product in a second type.
  Args:
    x, y (ArrayLike): Inputs
    x_dtype (Dtype): Data type of `x`
    y_dtype (Dtype): Data type of `y`
    out_dtype (Dtype): Output data type. Use `None` to skip output
      rounding, leaving it in the default type from `numpy`.
  Returns:
    :math:`x * y`
  """

  product = (
      to_dtype(x, x_dtype.name, x_dtype.round) *
      to_dtype(y, y_dtype.name, y_dtype.round))
  if out_dtype is None:
    return product  # no output rounding, leave as default type (np.float64)
  return to_dtype(product, out_dtype.name, out_dtype.round)

def add(x, y, x_dtype: Dtype, y_dtype: Dtype, out_dtype: Dtype = None):
  """Adds two numbers of one type and outputs sum in a second type.
  Args:
    x, y: Inputs
    x_dtype (Dtype): Data type of `x`
    y_dtype (Dtype): Data type of `y`
    out_dtype (Dtype): Output data type. Use `None` to skip output
      rounding, leaving it in the default type from `numpy`.
  Returns:
    :math:`x + y`
  """
  raw_sum = (
      to_dtype(x, x_dtype.name, x_dtype.round) +
      to_dtype(y, y_dtype.name, y_dtype.round))
  if out_dtype is None:
    return raw_sum
  return to_dtype(raw_sum, out_dtype.name, out_dtype.round)

def mult_add(x,
             y,
             z,
             x_dtype: Dtype,
             y_dtype: Dtype,
             z_dtype: Dtype,
             out_dtype: Dtype = None,
             fused=False):
  """Computes :math:`x y + z` using a multiplier with operands of one type
  and an adder with operands of another type with optional fused multiply-add.
  Args:
    x, y, z: Inputs
    x_dtype (Dtype): Treat `x` as this `Dtype`
    y_dtype (Dtype): Treat `y` as this `Dtype`
    z_dtype (Dtype): Treat `z` as this `Dtype`
    out_dtype (Dtype): Output data type.
    fused: If `True`, the product :math:`x y` is not rounded before being added
      to `z` to simulate fused multiply-add. Note that the code uses
      :py:func`pyfma.fma`, which may have problems under Windows
      (https://bugs.python.org/msg312480).
  Returns:
    :math:`x y + z`
  """
  if fused:
    return to_dtype(
        pyfma.fma(
            to_dtype(x, x_dtype.name, x_dtype.round),
            to_dtype(y, y_dtype.name, y_dtype.round),
            to_dtype(z, z_dtype.name, z_dtype.round),
        ), out_dtype.name, out_dtype.round)

  product = mult(x, y, x_dtype=x_dtype, y_dtype=y_dtype, out_dtype=out_dtype)
  raw_sum = product + to_dtype(z, z_dtype.name, z_dtype.round)
  if out_dtype is None:
    return raw_sum
  return to_dtype(raw_sum, out_dtype.name, out_dtype.round)

#    >>> x_fp16 = np.array([i for i in float_generator(11, range(-14, 16))])
#    >>> x_fp16.size, x_fp16[0], x_fp16[-1]
#    (30720, 6.103515625e-05, 65504.0)
#    >>> x_fp16 = np.array([i for i in float_generator(11, range(-14, 16), True)])
#    >>> x_fp16.size, x_fp16[0], x_fp16[-1]
#    (31743, 5.960464477539063e-08, 65504.0)
#    >>> x_bf16 = np.array([i for i in float_generator(8, range(-126, 128))])
#    >>> x_bf16.size, x_bf16[0], x_bf16[-1]
#    (32512, 1.1754943508222875e-38, 3.3895313892515355e+38
#    >>> x_bf16 = np.array([i for i in float_generator(8, range(-126, 128), True)])
#    >>> x_bf16.size, x_bf16[0], x_bf16[-1]
#    (32639, 9.183549615799121e-41, 3.3895313892515355e+38)

def float_generator(precision: int,
                    exponent_range: Iterable[int],
                    subnormals=True) -> Iterator[np.float64]:
  """Generates all positive floating-point numbers in the exponent range
  according to the precision.
  Examples:
    The following examples enumerate all normal floating-point
    numbers for the IEEE `binary16` format in the `numpy` array `x_fp16` and
    for the Google Brain Float `bfloat16` format in `x_bf16`.
    >>> x_fp16 = np.array([i for i in float_generator(11, range(-14, 16))])
    >>> x_fp16.size, x_fp16[0], x_fp16[-1]
    (31743, 5.960464477539063e-08, 65504.0)
    >>> x_bf16 = np.array([i for i in float_generator(8, range(-126, 128))])
    >>> x_bf16.size, x_bf16[0], x_bf16[-1]
    (32639, 9.183549615799121e-41, 3.3895313892515355e+38)
  Args:
    precison: Number of bits of precision in the floating-point format of
      interest, which is the number of bits in the trailing signficand field
      plus one for the hidden leading bit. For instance, the IEEE `binary32`
      format has 24 bits of precision and the `binary16` format has 11.
    exponent_range: An iterable in the form `range(exp_min, exp_max+1)`. For
      instance, the IEEE `binary16` exponent range is `range(-14, 16)` and
      the `binary32` exponent range is `range(-126, 128)`.
    subnormals: Outputs subnormals iff `True`
  Yields:
    Every floating-point number of the requested precision and
    exponent range starting with the lowest number.
  """
  if subnormals:
    for trailing_significand_int in range(1, 2**(precision - 1)):  # subnormals
      yield np.ldexp(trailing_significand_int / (2**precision),
                     exponent_range[0] + 1)
  for exponent in exponent_range:
    for trailing_significand_int in range(2**(precision - 1), 2**precision):
      yield np.ldexp(trailing_significand_int / (2**precision), exponent + 1)

def block_float(x, precision, block_size, block_axis):
  """Forces a block of floating-point numbers to share an exponent, which is
  the maximum exponent of the numbers in the block. Numbers with non-maximum
  exponents lose precision and may become zero.
  Args:
    x (ArrayLike): Input array
    block_size (int): Every consecutive `block_size` numbers along `block_axis`
      share an exponent.
    block_axis (int): Use this axis in `x` for grouping adjacent numbers into
      blocks.
  Returns:
    An array of the same shape as `x` as block-floating point numbers.
    Blocks of `block_size` numbers use the same exponent.
  """
  new_shape = list(x.shape) + [block_size]
  new_shape[block_axis] = -1  # infer x.shape[block_axis]//block_size
  frac, exp = np.frexp(x.reshape(new_shape))
  exp_max = exp.max(axis=len(x.shape))
  exp_diff = np.broadcast_to(np.expand_dims(exp_max, axis=-1), exp.shape) - exp
  new_frac = (
      np.rint(frac * np.power(2.0, precision - exp_diff)) *
      np.power(2.0, -precision + exp_diff))
  return np.ldexp(new_frac, exp).reshape(x.shape)

def diff_approx_func_unary(func,
                           ref_func,
                           x,
                           x_dtype,
                           func_kwargs=None,
                           ref_func_kwargs=None):
  """Compare differences between two functions"""
  x = to_dtype(x, x_dtype.name, x_dtype.round)
  func_y = func(x, **func_kwargs) if func_kwargs else func(x)
  ref_func_y = ref_func(x, **
                        ref_func_kwargs) if ref_func_kwargs else ref_func(x)
  diff = func_y - ref_func_y
  max_rel_diff = (diff / np.abs(ref_func_y)).max()
  max_abs_diff = (np.abs(diff)).max()
  return (func_y, ref_func_y, max_rel_diff, max_abs_diff)
