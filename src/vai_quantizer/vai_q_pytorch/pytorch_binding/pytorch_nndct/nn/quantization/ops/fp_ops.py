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

import torch

from pytorch_nndct.utils.hw_dtype import fp32
from pytorch_nndct.nn.quantization.ops import quantize_ops

def get_exponent(tensor):
  with torch.no_grad():
    t = torch.nan_to_num(tensor, nan=0, posinf=0, neginf=0)
    '''
    smallest positive subnormal, largest subnormal, smallest positive normal, largest normal
    >>> t = torch.tensor([0, 1.4012984643*10**-45, 1.1754942107*10**-38, 1.1754943508 *10**-38, 3.4028234664*10**38])
    >>> torch.frexp(t)
    torch.return_types.frexp(
    mantissa=tensor([0.0000, 0.5000, 1.0000, 0.5000, 1.0000]),
    exponent=tensor([0, -148, -126, -125,  128], dtype=torch.int32))
    '''
    _, exp = torch.frexp(t)
    # The exponent of subnormal is less than fp32.emin
    exp[(t==0) | (exp < fp32.emin)] = fp32.emin
  return exp - 1.0

def cast_to_fp(tensor, exp_bias, m_bits, round_mode, min_val, max_val):
  exp = get_exponent(tensor)
  # 2**(1 - exp_bias - m_bits) is the smallest representable value
  scale = torch.pow(2.0, torch.clamp(exp, 1 - exp_bias, fp32.emax + 1) - m_bits)
  return quantize_ops.quantize(tensor, scale, round_mode, min_val, max_val)
