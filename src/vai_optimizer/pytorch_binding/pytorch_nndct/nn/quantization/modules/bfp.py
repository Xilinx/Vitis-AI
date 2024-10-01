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

from pytorch_nndct.nn.quantization.ops import bfp_ops

class BFPQuantizer(nn.Module):

  def __init__(self, bitwidth, block_size, axis, rounding_mode='round_even'):
    super(BFPQuantizer, self).__init__()

    self.bitwidth = bitwidth
    self.block_size = block_size
    self.axis = axis
    self.rounding_mode = rounding_mode

    self._forward_fn = bfp_ops.quantize_to_bfp

  def forward(self, input):
    return self._forward_fn(input, self.bitwidth, self.block_size, self.axis,
                            self.rounding_mode)

  def extra_repr(self):
    return 'bitwidth={}, block_size={}, axis={}, rounding_mode={}'.format(
        self.bitwidth, self.block_size, self.axis, self.rounding_mode)

class BFPPrimeQuantizer(BFPQuantizer):

  def __init__(self, bitwidth, block_size, axis, rounding_mode='round_even'):
    super(BFPPrimeQuantizer, self).__init__(bitwidth, block_size, axis,
                                            rounding_mode)

    self._forward_fn = bfp_ops.quantize_to_bfp_prime

class BFPPrimeSharedQuantizer(BFPPrimeQuantizer):

  def __init__(self,
               bitwidth,
               block_size,
               sub_block_size,
               sub_block_shift_bits,
               axis,
               rounding_mode='round_to_nearest'):
    super(BFPPrimeSharedQuantizer, self).__init__(bitwidth, block_size, axis,
                                                  rounding_mode)
    self.sub_block_size = sub_block_size
    self.sub_block_shift_bits = sub_block_shift_bits
    self._forward_fn = bfp_ops.quantize_to_bfp_prime_shared

  def forward(self, input):
    return self._forward_fn(input, self.bitwidth, self.block_size,
                            self.sub_block_size, self.sub_block_shift_bits,
                            self.axis, self.rounding_mode)

  def extra_repr(self):
    return ('bitwidth={}, block_size={}, sub_block_size={}, '
            'sub_block_shift_bits={}, axis={}, rounding_mode={}').format(
                self.bitwidth, self.block_size, self.sub_block_size,
                self.sub_block_shift_bits, self.axis, self.rounding_mode)
