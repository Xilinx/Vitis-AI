

#
# Copyright 2019 Xilinx Inc.
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

from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
__all__ = ['Interpolate']

class deephi_Interpolate(torch.nn.Module):

  def __init__(self, *args, **kwards):
    super(deephi_Interpolate, self).__init__(*args, **kwards)
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()

  def forward(self,
              input,
              size=None,
              scale_factor=None,
              mode='nearest',
              align_corners=None):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input],
    )

    output = torch.nn.functional.interpolate(input, size, scale_factor, mode,
                                             align_corners)

    [output] = post_quant_process(self.node, [output])

    return output
  
@py_utils.register_quant_op
def Interpolate(*args, **kwargs):
  return deephi_Interpolate(*args, **kwargs)
