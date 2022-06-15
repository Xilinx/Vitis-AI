

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

import math
import torch
import numpy as np
from cmath import exp

from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors
from nndct_shared.utils import NndctOption
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS
from .sigmoid_table import *
from .fix_ops import NndctSoftmaxExpApproximate, NndctSoftmaxLOD, NndctSoftmaxSimulationPart1, NndctSoftmaxSimulationPart2
import pytorch_nndct.utils as py_utils

__all__ = ['Softmax']

class deephi_Softmax(torch.nn.modules.Softmax):
  r"""DeePhi Softmax operation"""

  def __init__(self, dim = None):
    super(deephi_Softmax, self).__init__()
    self.dim = dim
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, input):
    if self.quant_mode > 0:
      if self.node.in_quant_part:
        if NndctOption.nndct_softmax_sim.value == 1:
          # Method 1: Hardware PL Softmax
          qinput = quantize_tensors([input], self.node, tensor_type='input')[0]

          x_max = torch.max(qinput, dim = self.dim, keepdim = True).values
          Exp_sum_appr = 0.0
          softmax_sum = 0.0 
          softmax_appr_sum = 0.0
          
          uvi = 47274 / math.pow(2,15) * (qinput - x_max)
          exp_appr = torch.empty_like(uvi)
          NndctSoftmaxExpApproximate(uvi, exp_appr)
          
          exp_appr = torch.round(exp_appr*10**5)
          exp_appr = exp_appr/(10**5)
          Exp_sum_appr = torch.sum(exp_appr, dim = self.dim, keepdim = True)  

          F = Exp_sum_appr
          w = torch.empty_like(F)
          NndctSoftmaxLOD(F, w)
          m = F/(2**w) 

          lnF = torch.round((22713/(2**15))*(m-1+w)*10**5)/10**5
          uvi = 47274 / (2**15) * (qinput - x_max - lnF)
          exp_appr = torch.empty_like(uvi)
          NndctSoftmaxExpApproximate(uvi, exp_appr)
          exp_appr = torch.round(exp_appr*10**5)/10**5
          output = exp_appr
          
          output = quantize_tensors([output], self.node)[0]

        elif NndctOption.nndct_softmax_sim.value == 2:
          # Method 2: Hardware PL Softmax
          qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
          x_max = torch.max(qinput, dim = self.dim, keepdim = True).values
          qinput = qinput - x_max
          
          exp_appr = torch.empty_like(qinput)
          NndctSoftmaxSimulationPart1(qinput, exp_appr)
          sum = torch.sum(exp_appr, dim = self.dim, keepdim = True)

          sum1 = torch.empty_like(sum)
          NndctSoftmaxSimulationPart2(sum, sum1)
          output = (exp_appr*sum1).bfloat16().float()
          output = quantize_tensors([output], self.node)[0]
        else:
          output = super().forward(input)
      else:
        output = super().forward(input)
    else:
      output = super().forward(input)
    return output


@py_utils.register_quant_op
def Softmax(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Softmax(*args, **kwargs)
  return deephi_Softmax(*args, **kwargs)
