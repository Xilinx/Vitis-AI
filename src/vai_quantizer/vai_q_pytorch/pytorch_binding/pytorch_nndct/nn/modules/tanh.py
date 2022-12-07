

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
from .tanh_table import *
from .fix_ops import NndctTanhTableLookup, NndctTanhSimulation, NndctTanhTableLookupAIE2
import pytorch_nndct.utils as py_utils
__all__ = ['Tanh']

TANH_TABLE = deephi_tanh_table()

class deephi_Tanh(torch.nn.modules.Tanh):
  r"""DeePhi Tanh operation"""

  def __init__(self):
    super(deephi_Tanh, self).__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, input):
    if self.quant_mode == 0 or (not self.node.in_quant_part):
      return super().forward(input)

    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]

    if (NndctOption.nndct_quant_off.value or
        self.quantizer is None or
        self.quantizer.exporting or
        NndctOption.nndct_cv_app.value):
      # Method 0: quant input and output (for CV)
      output = super().forward(qinput)
      output = quantize_tensors([output], self.node)[0]

    else:
      output = torch.empty_like(qinput)
      input_name = self.node.in_nodes[0]
      input_node = self.quantizer.configer.get_Nndctnode(input_name)
      if not self.quantizer.configer.node_output_quantizable(input_node):
        input_name = input_node.in_nodes[0]

      fragpos = self.quantizer.get_quant_config(input_name, False)[1]
      # Method 1: Simulation AIE with 16 bw (for RNNT)
      if NndctOption.nndct_op_tanh_sigmoid_mode.value == "simulation":
        NndctTanhSimulation(input, output, fragpos)
        output = quantize_tensors([output], self.node)[0]
      # Method 2: Table Look up for AIE2 with 16 bw (based on LUT)
      elif NndctOption.nndct_op_tanh_sigmoid_mode.value == "aie2_lut_16bw" or NndctOption.nndct_ip_asr.value:
        NndctTanhTableLookupAIE2(qinput, output, fragpos)
        output = quantize_tensors([output], self.node)[0]
      # Method 3: Table Look up for FPGA with 16 bw
      else:
        quant_device = qinput.device
        Ttable = TANH_TABLE.table.to(qinput.dtype).to(quant_device)
        output = output.to(quant_device)
        NndctTanhTableLookup(input,
                             Ttable,
                             output,
                             fragpos)
        bnfp = self.quantizer.get_quant_config(input_name, False)
        bnfp[1] = 15
        self.quantizer.set_quant_config(self.node.name, bnfp)

    return output


@py_utils.register_quant_op
def Tanh(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Tanh(*args, **kwargs)
  return deephi_Tanh(*args, **kwargs)

