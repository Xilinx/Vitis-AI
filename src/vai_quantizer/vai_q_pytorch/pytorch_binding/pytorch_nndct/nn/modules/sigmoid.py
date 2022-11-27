

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
import numpy as np
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors
from nndct_shared.utils import NndctOption
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS
from .sigmoid_table import *
from .fix_ops import NndctSigmoidTableLookup, NndctSigmoidSimulation
import pytorch_nndct.utils as py_utils

__all__ = ['Sigmoid']

SIGMOID_TABLE = deephi_sigmoid_table()

class deephi_Sigmoid(torch.nn.modules.Sigmoid):
  r"""DeePhi Sigmoid operation"""

  def __init__(self):
    super(deephi_Sigmoid, self).__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, input):

    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]

    if NndctOption.nndct_quant_off.value or NndctOption.nndct_cv_app.value:
      output = super().forward(qinput)
      # quantize output
      output = quantize_tensors([output], self.node)[0]
    elif self.quant_mode > 0:
      output = torch.empty_like(qinput)
      if NndctOption.nndct_tanh_sigmoid_sim.value > 0:
        NndctSigmoidSimulation(qinput, output)
        output = quantize_tensors([output], self.node)[0]
      else:
        input_name = self.node.in_nodes[0]
        fragpos = self.quantizer.get_quant_config(input_name, False)[1]
        quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
        Ttable = SIGMOID_TABLE.table.to(quant_device)
        output = output.to(quant_device)
        NndctSigmoidTableLookup(input,
                                Ttable,
                                output,
                                fragpos)
    else:
      output = super().forward(qinput)

    return output


@py_utils.register_quant_op
def Sigmoid(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Sigmoid(*args, **kwargs)
  return deephi_Sigmoid(*args, **kwargs)
