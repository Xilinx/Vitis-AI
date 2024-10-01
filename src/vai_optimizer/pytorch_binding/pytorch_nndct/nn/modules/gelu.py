

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

import os
import re
import torch
from torch.autograd import Variable
import math
import numpy as np

from nndct_shared.utils import NndctOption, NndctScreenLogger, create_work_dir
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors 
from .quant_noise import eval_qnoise
import pytorch_nndct.utils as py_utils
import torch.nn.functional as F
from pytorch_nndct.utils.torch_utils import CmpFlag, compare_torch_version
__all__ = ['GELU']

class deephi_GELU(torch.nn.GELU):
  r"""DeePhi GELU operation, support float and double"""

  def __init__(self, approximate=None):
    if compare_torch_version(CmpFlag.GREATER_EQUAL, "1.12.0"):
      super(deephi_GELU, self).__init__(approximate)
    else:
      super(deephi_GELU, self).__init__()
      self.approximate = approximate 
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()

  def gelu(self, x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

  def forward(self, input):
    if self.quant_mode <= 0 or (not self.node.in_quant_part) or NndctOption.nndct_gemm88.value:
      return super().forward(input)

    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]

    if NndctOption.nndct_quant_off.value or self.quantizer.exporting:
      output = super().forward(qinput)

    else:
      # Method 1: Dynamic table look up with 8 bw
      if NndctOption.nndct_op_gelu_mode.value == "dynamic_table" or NndctOption.nndct_ip_v70_bert.value:
        output = self.gelu(qinput)

      # Method 0: Quant input and output of Softmax  
      elif compare_torch_version(CmpFlag.GREATER_EQUAL, "1.12.0"):

        output = F.gelu(qinput, approximate=self.approximate)
      else:
        output = F.gelu(qinput)
    
    output = quantize_tensors([output], self.node)[0]
    return output

@py_utils.register_quant_op
def GELU(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.GELU(*args, **kwargs)
  return deephi_GELU(*args, **kwargs)
