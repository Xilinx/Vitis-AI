

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
from nndct_shared.quantization import maybe_get_quantizer, process_inputs_and_params, post_quant_process
import pytorch_nndct.utils as py_utils
__all__ = ['Mean']

class deephi_Mean(torch.nn.Module):
  r"""DeePhi Concat operation"""

  def __init__(self, *args, **kwargs):
    super(deephi_Mean, self).__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, input, dim, keepdim):
    input, _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=input)
    output = torch.mean(input, dim, keepdim)
    [output] = post_quant_process(self.node, [output])

    return output
  
  
@py_utils.register_quant_op
def Mean(*args, **kwargs):
  #quant_mode,_ = maybe_get_quantizer()
  #if quant_mode==None:
  #    return
  return deephi_Mean(*args, **kwargs)
