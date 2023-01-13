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

  def __init__(self,
               bitwidth,
               axis,
               round_mode='round_even',
               tile_size=8,
               is_prime=False,
               memory_format='channels_first',
               epsilon=torch.pow(torch.tensor(2.0), -23)):
    super(BFPQuantizer, self).__init__()

    self.bitwidth = bitwidth
    self.axis = axis
    self.round_mode = round_mode
    self.tile_size = tile_size
    self.epsilon = epsilon

    self._forward_fn = bfp_ops.quantize_to_bfp_prime if is_prime else bfp_ops.quantize_to_bfp

  def forward(self, input):
    return self._forward_fn(input, self.bitwidth, self.tile_size,
                            self.axis, self.round_mode, self.epsilon)

  def extra_repr(self):
    return 'bitwidth={}, tile_size={}, axis={}, round_mode={}, epsilon={}'.format(
        self.bitwidth, self.tile_size, self.axis, self.round_mode, self.epsilon)
