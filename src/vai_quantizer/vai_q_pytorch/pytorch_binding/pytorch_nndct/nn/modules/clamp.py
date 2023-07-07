

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
from torch.autograd import Variable
import math

from nndct_shared.utils import NndctOption, NndctScreenLogger
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors 
from .quant_noise import eval_qnoise
import pytorch_nndct.utils as py_utils

__all__ = ['Clamp']

class deephi_clamp(torch.nn.Module):
  r"""DeePhi clamp operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_clamp, self).__init__(*args, **kwards)
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, input, min=None, max=None):
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    qmin, qmax = None, None
    if min is not None:
        min = torch.tensor(min).to(input.device)
        qmin = quantize_tensors([min], self.node, tensor_names=[self.params_name[0]], tensor_type='param')[0]
    if max is not None:
        max = torch.tensor(max).to(input.device)
        qmax = quantize_tensors([max], self.node, tensor_names=[self.params_name[1]], tensor_type='param')[0]
    output = torch.clamp(input=qinput, min=qmin, max=qmax) 
    output = quantize_tensors([output], self.node)[0]
    return output

    # # quantize parameters
    # qweight = None
    # inplace = (NndctOption.nndct_quant_off.value or 
    #     self.quantizer is not None and self.quantizer.inplace)

    # # quantize input tensor
    # qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    # # output = torch.clamp(input=qinput, min=min, max=max)
    # qmin, qmax = quantize_tensors(torch.tensor([min, max]), self.node, tensor_type='input')
    # output = torch.clamp(input=qinput, min=qmin, max=qmax) 
    # output = quantize_tensors([output], self.node)[0]

    # return output

@py_utils.register_quant_op
def Clamp(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.clamp(*args, **kwargs)
  return deephi_clamp(*args, **kwargs)
