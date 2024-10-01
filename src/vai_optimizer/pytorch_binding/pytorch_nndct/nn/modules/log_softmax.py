
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

from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors
from nndct_shared.utils import NndctOption
from .fix_ops import NndctExpApprAIE2, NndctLogSoftmaxFastLn, NndctLogSoftmaxSub
import pytorch_nndct.utils as py_utils

__all__ = ['LogSoftmax']

class deephi_LogSoftmax(torch.nn.modules.LogSoftmax):
  r"""DeePhi LogSoftmax operation"""

  def __init__(self, dim = None):
    super(deephi_LogSoftmax, self).__init__()
    self.dim = dim
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, input):
    if self.quant_mode == 0 or (not self.node.in_quant_part):
      return super().forward(input)

    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]

    if (NndctOption.nndct_quant_off.value or
        self.quantizer is None or
        self.quantizer.exporting):
      # Method 0: quant input and output 
      output = super().forward(qinput)
      output = quantize_tensors([output], self.node)[0]

    else:
      # Method: Table Look up for AIE2 (based on LUT) with 16 bw
      if NndctOption.nndct_op_softmax_mode.value == "aie2_lut_16bw":
        input_name = self.node.in_nodes[0]
        input_node = self.quantizer.configer.get_Nndctnode(input_name)
        if not self.quantizer.configer.node_output_quantizable(input_node):
          input_name = input_node.in_nodes[0]
        
        bw = self.quantizer.get_quant_config(input_name, False)[0]
        fragpos = self.quantizer.get_quant_config(input_name, False)[1]
        
        if bw == 8 and fragpos < 2:
          if fragpos < 2:
            qinput_max = torch.max(qinput, dim = self.dim, keepdim = True).values
            qinput -= qinput_max
          else:
            qinput -= 31.75
        else: # bw == 16
          if fragpos < 10:
            qinput_max = torch.max(qinput)
            qinput -= qinput_max
          else:
            qinput -= 32

        qinput_exp = torch.empty_like(input)
        NndctExpApprAIE2(qinput, qinput_exp, bw)

        exp_sum = torch.sum(qinput_exp, dim = self.dim, keepdim = True)
        ln_sum = torch.empty_like(exp_sum)
        NndctLogSoftmaxFastLn(exp_sum, ln_sum)
        output = torch.empty_like(input)
        NndctLogSoftmaxSub(qinput, output, ln_sum)
        output = quantize_tensors([output], self.node)[0]
      else:
        output = super().forward(qinput)
        output = quantize_tensors([output], self.node)[0]

    return output


@py_utils.register_quant_op
def LogSoftmax(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.LogSoftmax(*args, **kwargs)
  return deephi_LogSoftmax(*args, **kwargs)
