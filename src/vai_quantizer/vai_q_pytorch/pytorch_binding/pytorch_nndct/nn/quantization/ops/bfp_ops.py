# Copyright 2022 Xilinx Inc.
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
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_nndct.nn.quantization.ops import quantize_ops
from pytorch_nndct.utils import onnx_utils

def _get_exponent_v1(tensor, epsilon=2**-23):
  t = tensor.abs()
  # we use fp32's 1.mantissa_bits
  max_t, _ = t.max(t.dim() - 1, keepdim=True)
  max_exp = (max_t + epsilon).log2().floor()
  t_exp = (t + epsilon).log2().floor()
  return max_exp, t_exp

def _get_exponent_v2(tensor, epsilon=2**-23):
  _, exp = torch.frexp(tensor)
  # we use fp32's 1.mantissa_bits
  max_exp, _ = exp.max(exp.dim() - 1, keepdim=True)
  return max_exp - 1, 1

def _get_exponent_v3(tensor, epsilon=2**-23):
  tensor_shape = list(tensor.shape)
  exponent = torch.ops.vai.calculate_shared_exponent(tensor, tensor_shape[-1])
  tensor_shape[-1] = 1
  return exponent.reshape(tensor_shape), 1

_get_exponent = _get_exponent_v2

def _min_max_at_exp(exp, bit_width):
  # sign bits: 1, exponent bits: 8, no implicit leading 1
  mantissa_bits = bit_width - 9
  # The min/max representable value with exp
  # x = {sign, exp, mant} = {0 01001101 0000001}
  # e = int(01001101) - 127
  # M = int(0.000001) = 2^(-6)
  # x = (-1)^sign*2^e*M
  #   = 2^e * 2^(-6) = 2^(e - 6)
  min_v = torch.pow(2.0, exp - (mantissa_bits - 1))
  max_v = torch.pow(2.0, exp + 1) - min_v
  return min_v, max_v

def transform_to_block_wise(input, block_size=8, axis=1):
  """Transform input tensor to block-wised format (i.e. the block in last
  dimension) at given asix with paddings if needed.

    For example, given a input tensor with shape [N, C, H, W], axis = 1.
    The tensor will be transposed to [N*W*H, L, B], where B equals to
    `block_size`. The channels will be padded with zeros before tranpose to
    make it divisble by `block_size`.

    Args:
      input: Input tensor.
      block_size: The number of tensors in a block.
      axis: The axis of input tensor to be transposed.

    Returns:
      A transposed and padded tensor in block-wised format.
  """

  input_shape = tuple(input.shape)
  input_dims = len(input_shape)
  assert axis < input_dims
  # [N, C, H, W] -> [N, W, H, C] -> [N*W*H, C]
  input = input.transpose(axis, input_dims - 1)
  input = input.reshape((-1, input_shape[axis])).contiguous()

  padded_channels = block_size - input_shape[axis] % block_size
  if padded_channels != block_size:
    input = F.pad(input, (0, padded_channels))

  # [N*W*H, C] -> [N*W*H, L, B]
  return torch.reshape(input, (input.shape[0], -1, block_size)).contiguous()

def transform_block_to_shape(input, shape, axis=1):
  """Transform block-wised tensor to given `shape` and remove padded blocks
  if it has.

    For example, given a block-wised tensor with shape [N*W*H, L, B],
    shape = [N, H, W, C] and axis = 1. The function first reshape it to
    [N*W*H, L*B] and remove padded blocks to [N*W*H, C], then transform it
    given shape [N, C, H, W].

    Args:
      input: A tensor in block-wised format.
      shape: The shape of the output tensor.
      axis: The axis where the channels is located in the `shape`.

    Returns:
      A transposed and de-padded tensor with given `shape`.
  """

  # [N*W*H, L, B] -> [N*H*W, L*B] -> [N*W*H, C]
  input = torch.reshape(input, (input.shape[0], -1)).contiguous()
  input = input[:, :shape[axis]]

  shape = list(shape)
  shape[axis], shape[-1] = shape[-1], shape[axis]
  # [N*W*H, C] -> [N, W, H, C] -> [N, C, H, W]
  return input.reshape(shape).transpose(axis, len(shape) - 1).contiguous()

def transpose_to_block_wise(input, block_size=8, axis=1):
  input_shape = tuple(input.shape)
  assert axis < len(input_shape)
  # [N, C, H, W] -> [C, N, H, W] -> [C, N*H*W]
  input = input.transpose(axis, 0).reshape((input_shape[axis], -1)).contiguous()

  padded_channels = block_size - input_shape[axis] % block_size
  if padded_channels != block_size:
    input = F.pad(input, (0, 0, 0, padded_channels))

  # [C', N**H*W] -> [L, B, N*H*W] -> [B, N*H*W, L]
  return input.reshape(-1, block_size, input.shape[-1]).permute(1, 2,
                                                                0).contiguous()

def transpose_block_to_shape(input, shape, axis=1):
  assert input.shape[0] * input.shape[-1] >= shape[axis]
  ## [B, N*H*W, L] -> [L, B, N*H*W] -> [C', N*H*W] -> [C, N*W*H]
  input = input.permute(2, 0, 1).reshape((-1, input.shape[1])).contiguous()
  input = input[:shape[axis], :]

  shape = list(shape)
  shape[axis], shape[0] = shape[0], shape[axis]
  # [C, N*H*W] -> [C, N, H, W] -> [N, C, H, W]
  return input.reshape(shape).transpose(0, axis).contiguous()

def pad_to_block_last(tensor, block_size=8, axis=1):
  """Transpose input tensor to block-wised format (i.e. the block in last
  dimension) by given axis and pad zeros if needed.

    For example, given a input tensor with shape [N, C, H, W], axis = 1.
    The tensor will be transposed to [N, W, H, C], and then the last dimension
    will be padded with zeros to make it divisble by block_size.

    Args:
      tensor: Input tensor.
      block_size: The number of elements in a block.
      axis: The axis of input tensor to be transposed.

    Returns:
      A padded tensor in block-wised format.
  """

  shape = tensor.shape
  dims = tensor.dim()
  if axis >= dims:
    raise ValueError('Tensor dimension is {}, but given axis is {}'.format(
        dims, axis))
  # [N, C, H, W] -> [N, W, H, C]
  tensor = tensor.transpose(axis, dims - 1).contiguous()
  padded_channels = block_size - shape[axis] % block_size
  if padded_channels != block_size:
    tensor = F.pad(tensor, (0, padded_channels))
  return tensor

def depad_and_transpose(tensor, shape, axis=1):
  """Remove paddings from a block-wised tensor and transpose it to given
     shape.

    Args:
      tensor: A tensor in block-wised format.
      shape: The shape of the output tensor.
      axis: The axis where the channels is located in the `shape`.

    Returns:
      A de-padded tensor with given shape.
  """

  # [N, H, W, C'] -> [N, H, W, C] -> [N, C, H, W]
  dims = tensor.dim()
  tensor = tensor[..., :shape[axis]]
  return tensor.transpose(axis, dims - 1).contiguous()

class BFPQuantize(torch.autograd.Function):

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, None, None, None

class BFPQuantizeV2(BFPQuantize):

  @staticmethod
  def forward(ctx, t, bit_width, block_size, round_mode='round_even'):
    out = torch.empty_like(t)
    return torch.ops.vai.to_bfp(t, bit_width, block_size, out)

class BFPQuantizeV3(BFPQuantize):

  @staticmethod
  def forward(ctx, t, bit_width, block_size, round_mode='round_even'):
    out = torch.empty_like(t)
    return torch.ops.vai.to_bfp_v2(t, bit_width, block_size, out)

def _to_bfp_v1(t,
               bit_width,
               round_mode='round_even',
               epsilon=torch.pow(torch.tensor(2.0), -23)):
  max_exp, _ = _get_exponent(t, epsilon)
  interval, max_v = _min_max_at_exp(max_exp, bit_width)
  return quantize_ops.quantize(t, interval, round_mode, -max_v, max_v)

if onnx_utils.support_onnx_export():
  _to_bfp_op = torch.ops.vai.to_bfp
else:
  from pytorch_nndct.nn.load_kernels import nndct_kernels
  _to_bfp_op = nndct_kernels.to_bfp

def _to_bfp_v2(t,
               bit_width,
               block_size,
               round_mode='round_even'):
  out = torch.empty_like(t)
  #return _to_bfp_op(t, 8, bit_width, out)
  return torch.ops.vai.to_bfp(t, bit_width, block_size, out)

def _to_bfp_v3(t,
               bit_width,
               block_size,
               round_mode='round_even'):
  out = torch.empty_like(t)
  return torch.ops.vai.to_bfp_v2(t, bit_width, block_size, out)

def _to_bfp_prime(t,
                  bit_width,
                  round_mode='round_even',
                  epsilon=torch.pow(torch.tensor(2.0), -23)):
  max_exp, t_exp = _get_exponent(t, epsilon)
  # offset = max_axp - current_exp
  # prime_bit = 0 if offset >= 1 else 1
  # shared_exp = max_exp - 1 if prime_bit == 0 elese max_exp
  # offset greater or equal than one
  offset_ge_one = max_exp - t_exp >= 1
  shared_exp = offset_ge_one * (-1) + max_exp
  interval, max_v = _min_max_at_exp(shared_exp, bit_width)
  t = quantize_ops.quantize(t, interval, round_mode, -max_v, max_v)
  return t

# input [N, IC, W, H] or [N, IC] or [OC, IC, K, K], or [OC, IC]. IC is important
def quantize_to_bfp_v1(tensor,
                       bit_width,
                       block_size,
                       axis,
                       round_mode,
                       epsilon=torch.pow(torch.tensor(2.0), -23)):

  shape = tensor.shape
  tensor = transform_to_block_wise(tensor, block_size, axis)
  tensor = _to_bfp_v1(tensor, bit_width, round_mode, epsilon)
  return transform_block_to_shape(tensor, shape, axis)

# tensor [N, IC, W, H] or [N, IC] or [OC, IC, K, K], or [OC, IC]. IC is important
def quantize_to_bfp_v2(tensor,
                       bit_width,
                       block_size,
                       axis,
                       round_mode):
  shape = tensor.shape
  tensor = transpose_to_block_wise(tensor, block_size, axis)
  tensor = BFPQuantizeV2.apply(tensor, bit_width, block_size, round_mode)
  return transpose_block_to_shape(tensor, shape, axis)

def quantize_to_bfp_v3(tensor,
                       bit_width,
                       block_size,
                       axis,
                       round_mode):
  shape = tensor.shape
  tensor = pad_to_block_last(tensor, block_size, axis)
  tensor = BFPQuantizeV3.apply(tensor, bit_width, block_size, round_mode)
  return depad_and_transpose(tensor, shape, axis)

quantize_to_bfp = quantize_to_bfp_v3

def quantize_to_bfp_prime(tensor,
                          bit_width,
                          block_size,
                          round_mode,
                          axis,
                          epsilon=torch.pow(torch.tensor(2.0), -23)):

  shape = tensor.shape
  tensor = transform_to_block_wise(tensor, block_size, axis)
  tensor = _to_bfp_prime(tensor, bit_width, round_mode, epsilon)
  return transform_block_to_shape(tensor, shape, axis)
