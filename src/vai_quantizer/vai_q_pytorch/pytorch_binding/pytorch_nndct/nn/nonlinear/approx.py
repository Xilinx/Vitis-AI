import torch
import numpy as np

from .coefficient import get_sigmoid_positive_ploy_coeffcients, get_exp_poly_coeffcients, get_gelu_tanh_poly_coeffcients, get_tanh_positive_poly_coeffcients
from .hw_dtype import is_subnormal, is_normal

def mult_add(a_bf16, b_bf16, c_fp32):
  assert a_bf16.dtype == torch.bfloat16 and b_bf16.dtype == torch.bfloat16
  out = a_bf16.to(torch.float32) * b_bf16.to(torch.float32) + c_fp32
  return out

def ploy_HORNER_SCHEME(r, cs, degree):
  out = mult_add(r, cs[degree].to(torch.bfloat16), cs[degree - 1])
  for i in reversed(range(degree - 1)):
    out = mult_add(r, out.to(torch.bfloat16), cs[i])
  out = out.to(r.dtype)

  return out

def sigmoid_positive_approx(z, degree=4):
  rge = [-7.0, 7.0]
  zero_mask = (z < rge[0]).to(torch.bfloat16)
  one_mask = (z > rge[1]).to(torch.bfloat16)

  x_sgn, x = torch.sign(z), torch.clamp(torch.abs(z), 0, rge[1])
  m_zero_mask = (x_sgn < 0).to(torch.bfloat16)

  cs = get_sigmoid_positive_ploy_coeffcients(degree)
  coeffs = torch.from_numpy(cs).to(z.device)

  # out = coeffs[0] + z * (coeffs[1] + z * (coeffs[2] + z * coeffs[3]))
  out = ploy_HORNER_SCHEME(x, coeffs, degree)
  out = m_zero_mask - out * m_zero_mask + out * (1 - m_zero_mask)

  out = out * (1.0 - zero_mask - one_mask) + one_mask
  return out

def tanh_positive_approx(z, degree=4):
  assert z.dtype == torch.bfloat16
  rge = [-4.0, 4.0]
  m_one_mask = (z < rge[0]).to(torch.bfloat16)
  one_mask = (z > rge[1]).to(torch.bfloat16)

  x_sgn, x = torch.sign(z), torch.clamp(torch.abs(z), 0, rge[1])

  cs = get_tanh_positive_poly_coeffcients(degree=degree)
  coeffs = torch.from_numpy(cs).to(z.device)

  out = ploy_HORNER_SCHEME(x, coeffs, degree)
  out = out * x_sgn
  out = out * (1.0 - m_one_mask - one_mask) + one_mask + m_one_mask * (-1)

  return out

def tanh_with_exp_approx(z,
                         degree=3,
                         exp_table_size=1,
                         output_subnormals=False,
                         dtype=torch.bfloat16):
  assert z.dtype == dtype

  sign = torch.sign(z).to(dtype)

  x = torch.abs(z)
  out = -2.0 * x
  out = 1.0 + exp_approx_poly(
      out.to(torch.float32), exp_table_size=exp_table_size,
      degree=degree).to(dtype)
  out = reciprocal_approx_moroz(
      out.to(torch.float32), output_subnormals=output_subnormals).to(dtype)
  out = 2.0 * out - 1.0

  out = out * sign

  assert out.dtype == dtype
  return out

def tanh_with_exp_approx_lut(z, output_subnormals=False, dtype=torch.bfloat16):
  assert z.dtype == dtype

  sign = torch.sign(z).to(dtype)

  x = torch.abs(z)
  out = -2.0 * x
  out = 1.0 + exp_approx_lut(out.to(torch.bfloat16)).to(dtype)
  out = reciprocal_approx_moroz(
      out.to(torch.float32), output_subnormals=output_subnormals).to(dtype)
  out = 2.0 * out - 1.0

  out = out * sign

  assert out.dtype == dtype
  return out

LN2 = 0.6931471805599453

_AMD_FLOOR = np.floor
_AMD_FLOOR = torch.floor
_AMD_ROUND = torch.round

from distutils.version import LooseVersion
# torch do not have good support for bfloat16 operations prior to 1.8
_is_torch_ge_180 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')
if _is_torch_ge_180:
  _MAX_BFLOAT16 = torch.finfo(torch.bfloat16).max
  _MIN_BFLOAT16 = torch.finfo(torch.bfloat16).min

def ploy_HORNER_SCHEME(r, cs, degree):
  out = mult_add(r, cs[degree].to(torch.bfloat16), cs[degree - 1])
  for i in reversed(range(degree - 1)):
    out = mult_add(r, out.to(torch.bfloat16), cs[i])
  out = out.to(r.dtype)

  return out

# from https://github.com/amd/aocl-libm-ose/blob/master/src/optmized/exp.c
# input: fp32
# output: bfloat16
def exp_approx_poly(x, exp_table_size=1, degree=3, method='HORNER'):
  assert method == 'HORNER'
  assert x.dtype == torch.float32

  EXP_TABLE_SIZE = exp_table_size
  cs = get_exp_poly_coeffcients(exp_table_size, degree)
  cs = torch.from_numpy(cs).to(x.device)

  if EXP_TABLE_SIZE > 1:
    scaling_factor = EXP_TABLE_SIZE / np.log(2)
    n = _AMD_ROUND(x.to(torch.float32) * scaling_factor)
    m = _AMD_FLOOR((n / EXP_TABLE_SIZE).to(torch.float32))  #.to(torch.bfloat16)
  else:
    scaling_factor = x.to(torch.bfloat16) * (
        torch.scalar_tensor(EXP_TABLE_SIZE / LN2).to(torch.bfloat16))
    n = _AMD_ROUND(scaling_factor.to(torch.float32)).to(torch.bfloat16)
    m = _AMD_FLOOR((n / EXP_TABLE_SIZE).to(torch.float32)).to(torch.bfloat16)
  j = n - m * EXP_TABLE_SIZE  #n % EXP_TABLE_SIZE

  # exp = 2**( (EXP_TABLE_SIZE * m + j + f) / EXP_TABLE_SIZE)

  # (j/EXP_TABLE_SIZE) < 1.0: Look up table for 2**(j/EXP_TABLE_SIZE).
  # r = np.exp(f*(LN2/EXP_TABLE_SIZE)) < 1.0: Polynomial
  r = x - n.to(torch.float32) * (LN2 / EXP_TABLE_SIZE)
  # exp = 2**m * (2**(j/EXP_TABLE_SIZE)) * np.exp(r)
  if method == 'HORNER':
    # exp_x = ( 2**m * (2**(j/EXP_TABLE_SIZE)) * ploy_HORNER_SCHEME(r.to(torch.bfloat16), cs, degree) ).to(torch.bfloat16
    exp_x = (2**(j / EXP_TABLE_SIZE).to(torch.bfloat16)) * ploy_HORNER_SCHEME(
        r.to(torch.bfloat16), cs, degree)

    if EXP_TABLE_SIZE > 1:
      assert m.dtype == torch.float32 and exp_x.dtype == torch.bfloat16
    else:
      assert m.dtype == torch.bfloat16 and exp_x.dtype == torch.bfloat16
    exp_x = 2**m.to(torch.float32) * exp_x.to(torch.float32)

    exp_x = torch.clamp(exp_x, _MIN_BFLOAT16, _MAX_BFLOAT16).to(torch.bfloat16)
    assert exp_x.dtype == torch.bfloat16

  return exp_x

def exp_approx_lut(x, lg_table_depth=8):
  assert x.dtype == torch.bfloat16

  beta_t = np.exp2(lg_table_depth).astype(int)

  scaling_factor = beta_t / np.log(2)
  scaled_x = _AMD_ROUND(x.to(torch.float32) * scaling_factor)
  assert scaled_x.dtype == torch.float32

  quotient = _AMD_FLOOR(scaled_x / beta_t)
  remainder = scaled_x - quotient * beta_t  #scaled_x % beta_t

  # fraction, exponent = np.frexp(to_dtype(np.exp2(remainder/beta_t), out_dtype))
  # exponent = (exponent + quotient).astype(int)
  # y = np.ldexp(fraction, exponent)
  y = (2**quotient) * (2**(remainder / beta_t).to(torch.bfloat16))
  y = torch.clamp(y, _MIN_BFLOAT16, _MAX_BFLOAT16).to(torch.bfloat16)
  assert y.dtype == torch.bfloat16

  return y

def sigmoid_with_exp_approx(z,
                            degree=3,
                            exp_table_size=1,
                            output_subnormals=False,
                            dtype=torch.bfloat16):
  assert z.dtype == dtype

  z = z.to(torch.float32)
  out = 1.0 + exp_approx_poly(
      -z, exp_table_size=exp_table_size, degree=degree).to(dtype)
  out = reciprocal_approx_moroz(
      out.to(torch.float32), output_subnormals=output_subnormals)
  out = out.to(dtype)

  assert out.dtype == dtype
  return out

def sigmoid_with_exp_approx_lut(z,
                                output_subnormals=False,
                                dtype=torch.bfloat16):
  assert z.dtype == dtype

  z = z.to(torch.bfloat16)
  out = 1.0 + exp_approx_lut(-z).to(dtype)
  out = reciprocal_approx_moroz(
      out.to(torch.float32), output_subnormals=output_subnormals)
  out = out.to(dtype)

  assert out.dtype == dtype
  return out

def softmax_approx_poly(x, axis=-1, exp_table_size=1, degree=3):
  x = x - torch.max(x.to(torch.float32), axis, keepdims=True)[0].to(x.dtype)

  exp_x = exp_approx_poly(x, exp_table_size=exp_table_size, degree=degree)

  # NOTE: approximate: r = exp_x / torch.sum(exp_x, axis, keepdim=True)
  exp_x_sum = torch.sum(exp_x.to(torch.float32), axis, keepdim=True)
  exp_x_sum_reciprocal = reciprocal_approx_moroz(exp_x_sum)
  r = exp_x.to(torch.float32) * exp_x_sum_reciprocal
  r = r.to(torch.bfloat16)
  return r

def softmax_approx_lut(x, axis=-1, lg_table_depth=8):
  x = x - torch.max(x.to(torch.float32), axis, keepdims=True)[0].to(x.dtype)

  exp_x = exp_approx_lut(x, lg_table_depth=lg_table_depth)

  # NOTE: approximate: r = exp_x / torch.sum(exp_x, axis, keepdim=True)
  exp_x_sum = torch.sum(exp_x.to(torch.float32), axis, keepdim=True)
  exp_x_sum_reciprocal = reciprocal_approx_moroz(exp_x_sum)
  r = exp_x.to(torch.float32) * exp_x_sum_reciprocal
  r = r.to(torch.bfloat16)
  return r

def gelu_approx(x, degree=2):
  assert x.dtype == torch.bfloat16
  sqrt_2 = 1.4142

  rge = [-2.5, 2.5]
  zero_mask = (x < rge[0]).to(x.device).to(torch.bfloat16)
  y_e_x_mask = (x > rge[1]).to(x.device).to(torch.bfloat16)

  q = x
  q_sgn, q = torch.sign(q), torch.clamp(torch.abs(q), 0, rge[1])

  cs = get_gelu_tanh_poly_coeffcients(degree=degree)
  coeffs = torch.from_numpy(cs).to(x.device).to(x.device)

  out = ploy_HORNER_SCHEME(q, coeffs, degree)
  out = x * 0.5 * (1 + out * q_sgn)

  out = out * (1.0 - zero_mask - y_e_x_mask) + x * y_e_x_mask
  assert out.dtype == torch.bfloat16
  return out

def isqrt_approx_bfloat16(x, mantissa_bit=7, exponent_bit=8):
  # From fast inverse squre root.
  # gamma = 0.0450466 first order Taylor error.
  # magic_number = 3/2 * 2^mantissa_bit * (2^(exponent_bit-1) - 1 - gamma)
  # gamma = 0.0450466
  # magic_n = np.array(round(3.0/2.0 * 2.0**mantissa_bit * (2.0**(exponent_bit-1) - 1.0 - gamma)), dtype=np.int16)

  # From https://www.mdpi.com/1099-4300/23/1/86/pdf
  magic_n = np.round(2**mantissa_bit *
                     (3 * (2.0**(exponent_bit - 1) - 1.0) - 1) / 2 +
                     np.round(2**mantissa_bit *
                              (3.7315712401613957182292407381942955 - 2) / 4 -
                              0.5))
  magic_n = np.array(magic_n, dtype=np.int16)

  x2 = (x * 0.5).to(torch.bfloat16)
  number = x.to(torch.float32).cpu().detach().numpy()
  threehalfs = 1.5
  y = np.float32(number)

  i = y.view(np.int32)
  i = (magic_n - np.int32(i >> 17)) << 16
  y = i.view(np.float32)

  y = torch.from_numpy(y).to(x.device).to(x.dtype)

  out = (x2 * y).to(torch.bfloat16)
  out = (out * y).to(torch.bfloat16)
  out = (threehalfs - out).to(torch.bfloat16)
  out = (y * out).to(torch.bfloat16)
  # y = y * (threehalfs - (x2 * y * y))
  # y = y * (threehalfs - (x2 * y * y))
  return out

def isqrt_approx_quake(x, mantissa_bit=23, exponent_bit=8):
  # From fast inverse squre root.
  # gamma = 0.0450466 first order Taylor error.
  # magic_number = 3/2 * 2^mantissa_bit * (2^(exponent_bit-1) - 1 - gamma)
  gamma = 0.0450466
  magic_n = np.array(
      np.round(3.0 / 2.0 * 2.0**mantissa_bit *
               (2.0**(exponent_bit - 1) - 1.0 - gamma)),
      dtype=np.int32)

  # From https://www.mdpi.com/1099-4300/23/1/86/pdf
  # magic_n = np.round(2**mantissa_bit *(3* (2.0**(exponent_bit-1) - 1.0) - 1) /2 + np.round(2**mantissa_bit * (3.7315712401613957182292407381942955 - 2)/4 - 0.5))
  # magic_n = np.array(magic_n, dtype=np.int32)

  number = x.cpu().numpy()
  threehalfs = 1.5
  x2 = number * 0.5
  y = np.float32(number)

  i = y.view(np.int32)
  i = magic_n - np.int32(i >> 1)
  y = i.view(np.float32)

  y = y * (threehalfs - (x2 * y * y))
  y = y * (threehalfs - (x2 * y * y))
  return torch.from_numpy(y).to(x.device).to(x.dtype)

def isqrt_approx_walcyzk(x, mantissa_bit=23, exponent_bit=8):
  assert x.dtype == torch.float32
  # From fast inverse squre root.
  # gamma = 0.0450466 first order Taylor error.
  # magic_number = 3/2 * 2^mantissa_bit * (2^(exponent_bit-1) - 1 - gamma)
  # gamma = 0.0450466
  # magic_n = np.array(round(3.0/2.0 * 2.0**mantissa_bit * (2.0**(exponent_bit-1) - 1.0 - gamma)), dtype=np.int16)

  # From https://www.mdpi.com/1099-4300/23/1/86/pdf
  # magic_n = np.round(2**mantissa_bit *(3* (2.0**(exponent_bit-1) - 1.0) - 1) /2 + np.round(2**mantissa_bit * (3.7315712401613957182292407381942955 - 2)/4 - 0.5))
  magic_n = 0x5f376908  # np.array(magic_n, dtype=np.int32)

  number = x.cpu().numpy()
  threehalfs1 = 1.50087896
  threehalfs2 = 1.50000057
  x2 = number * 0.5
  y = np.float32(number)

  i = y.view(np.int32)
  i = magic_n - np.int32(i >> 1)
  y = i.view(np.float32)

  y = y * (threehalfs1 - (x2 * y * y))
  y = y * (threehalfs2 - (x2 * y * y))
  return torch.from_numpy(y).to(x.device).to(x.dtype)

def reciprocal_approx_isqrt(x,
                            input_dtype=torch.bfloat16,
                            output_dtype=torch.bfloat16,
                            isqrt_dtype=torch.bfloat16):
  assert x.dtype == input_dtype

  sign = torch.sign(x).to(output_dtype)
  out = torch.abs(x)

  if isqrt_dtype == torch.float32:
    out = isqrt_approx_walcyzk(out.to(isqrt_dtype))
  elif isqrt_dtype == torch.bfloat16:
    out = isqrt_approx_bfloat16(out.to(isqrt_dtype))
  else:
    raise NotImplementedError("only support isqrt fp32 and bfloat16.")

  out = out.to(output_dtype)
  out = out * out
  out = out * sign

  assert out.dtype == output_dtype
  return out

def reciprocal_approx_moroz(z, output_subnormals=False):
  assert z.dtype == torch.float32

  # @Ephrem (ephremw@xilinx.com)
  def okay_to_process(x):
    r"""Indicates which elements in the tensor are ready for reciprocal
        approximation.
        The FP32 exponent ranges from -126 to 127. The largest number whose reciprocal
        does not become subnormal is :math:`2^{126}` because :math:`2^{-126}`
        is the smallest normal number.
        This function returns `True` if `x` is normal and :math:`0 < |x| \le 2^{126}`.
        """
    okay_idx = np.logical_and(is_normal(x), x != 0)
    if output_subnormals:
      return okay_idx
    return np.logical_and(
        okay_idx,
        x <= np.exp2(
            min(
                # maxexp=128, smallest for overflow
                # pylint: disable=no-member
                np.abs(np.finfo(np.float32).minexp),
                np.abs(np.finfo(np.float32).maxexp - 2))))

  def mult_add_func(a, b, c):
    return a * b + c

  magic_n = np.int32(0x7eb53567)
  k1 = np.float32(1.9395974)
  k2 = np.float32(1.436142)
  magic = [magic_n, k1, k2]

  x = z.detach().cpu().numpy()

  frac, exponent = np.frexp(x)
  reduced_x = np.ldexp(frac, 1, dtype=np.float32)
  y_as_int = np.int32(magic[0]) - reduced_x.view(np.int32)
  y_as_fp32 = y_as_int.view(np.float32)

  y_as_fp32 = np.where(
      okay_to_process(x),
      # 1st (modified) Newton-Raphson
      mult_add_func(
          mult_add_func(magic[1], y_as_fp32, 0.0),
          mult_add_func(-reduced_x, y_as_fp32, np.float32(magic[2])), 0.0),
      y_as_fp32)
  # r = 1 - xy, y = ry + y # 2nd (classic) Newton-Raphson
  y_as_fp32 = np.where(
      okay_to_process(x),
      mult_add_func(y_as_fp32,
                    mult_add_func(y_as_fp32, -reduced_x, np.float32(1.0)),
                    y_as_fp32), y_as_fp32)

  out = np.where(
      np.logical_or(np.isfinite(x) is False, exponent > 126 + 1),
      np.copysign(np.float32(0), x),
      np.where(
          np.logical_or(is_subnormal(x), x == 0),
          np.copysign(np.float32(np.inf), x),
          np.ldexp(y_as_fp32, -exponent + 1).astype(np.float32)))

  out_tensor = torch.from_numpy(out).to(z.device).to(z.dtype)
  return out_tensor
