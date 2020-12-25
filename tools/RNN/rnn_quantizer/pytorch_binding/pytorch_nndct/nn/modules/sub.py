

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
__all__ = ['Sub']

class deephi_Sub(torch.nn.Module):

  def __init__(self):
    super(deephi_Sub, self).__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, input, other, alpha=1):
    [input, other], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input, other])
    output = torch.sub(input=input, other=other, alpha=alpha)
    [output] = post_quant_process(self.node, [output])
    return output

@py_utils.register_quant_op
def Sub(*args, **kwargs):
  return deephi_Sub(*args, **kwargs)

