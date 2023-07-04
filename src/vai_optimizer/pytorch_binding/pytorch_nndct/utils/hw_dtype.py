"""Data type conversion routines """
from typing import NamedTuple
from typing import Iterator, Iterable
import math
import numpy as np

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

class DType(NamedTuple):
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
    """Returns :py:type:`DType` from string name"""
    return DTYPES[self.name]

fp16 = DType('fp16', 'rn', -30, 15, 11)
"""FP16 shorthand"""
bf16 = DType('bf16', 'rn', -126, 127, 8)
"""Google Brain Float16 shorthand"""
tf32 = DType('tf32', 'rn', -126, 127, 11)
"""Tensor Float16 shorthand"""
fp32 = DType('fp32', 'rn', -126, 127, 24)
"""FP32 shorthand"""
fp64 = DType('fp64', 'rn', -1022, 1023, 53)
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
