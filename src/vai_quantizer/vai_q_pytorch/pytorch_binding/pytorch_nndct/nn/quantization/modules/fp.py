# Copyright 2023 Xilinx Inc.
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

import torch.nn as nn

from pytorch_nndct.nn.quantization.ops import fp_ops

class BFloat16Quantizer(nn.Module):

  def __init__(self):
    super(BFloat16Quantizer, self).__init__()

  def forward(self, x):
    return x.bfloat16().float()

class FP32Quantizer(nn.Module):

  def __init__(self):
    super(FP32Quantizer, self).__init__()

  def forward(self, x):
    return x.float()

class FP8Quantizer(nn.Module):
  '''
  Optimial bias mode: NaN/INF are represented as negative zero
  ==============================================================================
                 E4M3                           E5M2
  ------------------------------------------------------------------------------
  Exponent bias  8                              16
  Infinities     1.0000.000                     1.00000.00
  NaN            1.0000.000                     1.00000.00
  Zeros          0.0000.000                     0.00000.00
  Max normal     S.1111.111 = 1.875 * 2^7 = 240 S.11111.11 = 1.75 * 2^15 = 57344
  Min normal     S.0001.000 = 2^-7              S.00001.00 = 2^-15
  Max subnormal  S.0000.111 = 0.875 * 2^-7      S.00000.11 = 0.75 * 2^-15
  Min subnormal  S.0000.001 = 2^-10             S.00000.01 = 2^-17
  ------------------------------------------------------------------------------
  '''

  def __init__(self,
               exponent_bits,
               bias_mode='ieee',
               exponent_bias=None,
               round_mode='round_even'):
    super(FP8Quantizer, self).__init__()

    self.exponent_bits = exponent_bits
    self.round_mode = round_mode

    self.mantissa_bits = 8 - self.exponent_bits - 1
    self.bias_mode = bias_mode

    if bias_mode == 'ieee':
      self.max_exponent = 2**self.exponent_bits - 2
      self.exponent_bias = 2**(self.exponent_bits - 1) - 1
    elif bias_mode == 'optimial':
      self.max_exponent = 2**self.exponent_bits - 1
      self.exponent_bias = 2**(self.exponent_bits - 1)
    elif bias_mode == 'nvidia_arm_intel':
      self.max_exponent = 2**self.exponent_bits - 1
      self.exponent_bias = 2**(self.exponent_bits - 1) - 1
    else:
      raise NotImplementedError(
          'bias mode only support: ieee, optimial, nvidia_arm_intel, but got {}'
          .format(bias_mode))

    if exponent_bias is not None:
      if not isinstance(exponent_bias, int):
        raise ValueError(f'Exponent bias must int, but got {exponent_bias}')
      self.exponent_bias = exponent_bias

    if bias_mode == 'nvidia_arm_intel':
      # S.1111.110: 1 + 1 - 2^(1-m)
      self.max_normal = 2**(self.max_exponent - self.exponent_bias) * (
          2 - 2**(1 - self.mantissa_bits))
    else:
      # S.1111.111: 1 + 1 - 2^(-m)
      self.max_normal = 2**(self.max_exponent -
                            self.exponent_bias) * (2 - 2**(-self.mantissa_bits))

  def forward(self, input):
    return fp_ops.cast_to_fp(input, self.exponent_bias, self.mantissa_bits,
                             self.round_mode, -self.max_normal,
                             self.max_normal)

  def extra_repr(self):
    return f'exponent_bits={self.exponent_bits}, mantissa_bits={self.mantissa_bits}, round_mode={self.round_mode}, exponent_bias={self.exponent_bias}'
