

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
import torch.nn.functional as F

__all__ = ['PReLU']

class deephi_PReLU(torch.nn.PReLU):
  r"""DeePhi PReLU operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_PReLU, self).__init__(*args, **kwards)
    self.params_name = None
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.param_saved = False
    self.param_quantized = False

  def forward(self, input):
    # quantize parameters
    qweight = None
    inplace = (NndctOption.nndct_quant_off.value or 
        self.quantizer is not None and self.quantizer.inplace)
    if (not self.param_quantized):
      if inplace:
        _ = quantize_tensors(
            [self.weight],
            self.node,
            tensor_names = [self.params_name[0]],
            tensor_type = 'param')[0]
        qweight = self.weight
      else:
        qweight = quantize_tensors(
            [self.weight],
            self.node,
            tensor_names = [self.params_name[0]],
            tensor_type = 'param')[0]
      if not NndctOption.nndct_quant_off.value:
        self.param_quantized = True
    else:
      qweight = self.weight

    # quantize input tensor
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    output = torch.nn.functional.prelu(input = qinput, weight = qweight) 
    output = quantize_tensors([output], self.node)[0]

    return output

@py_utils.register_quant_op
def PReLU(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.PReLU(*args, **kwargs)
  return deephi_PReLU(*args, **kwargs)
