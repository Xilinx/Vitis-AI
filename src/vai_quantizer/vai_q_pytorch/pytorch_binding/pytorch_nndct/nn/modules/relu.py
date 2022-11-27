

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
from nndct_shared.utils import NndctOption
from nndct_shared.quantization import quantize_tensors 
import pytorch_nndct.utils as py_utils
__all__ = ['ReLU']

class deephi_ReLU(torch.nn.ReLU):
  r"""DeePhi ReLU operation"""

  def __init__(self, *args, **kwargs):
    super(deephi_ReLU, self).__init__(*args, **kwargs)
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, input):
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    output = super().forward(qinput)
    output = quantize_tensors([output], self.node)[0]
    return output
  
@py_utils.register_quant_op
def ReLU(*args, **kwargs):
  #quant_mode,_ = maybe_get_quantizer()
  #if quant_mode==None:
  #    return
  return deephi_ReLU(*args, **kwargs)
