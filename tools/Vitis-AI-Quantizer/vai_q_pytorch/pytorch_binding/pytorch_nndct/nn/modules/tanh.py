

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
from nndct_shared.quantization import maybe_get_quantizer, process_inputs_and_params, post_quant_process
from nndct_shared.utils import NndctOption
from .tanh_table import *
from .fix_ops import NndctTanhTableLookup
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

    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input])

    if NndctOption.nndct_quant_off.value:
      output = super().forward(input)
    elif self.quant_mode > 0:
      output = torch.empty_like(input)
      input_name = self.node.in_nodes[0]
      fragpos = self.quantizer.get_bnfp(input_name, False)[1]
      if(input.device == torch.device("cpu")):
        Ttable = TANH_TABLE.table.to(torch.device("cpu"))
        output = output.to(torch.device("cpu"))
      else:
        Ttable = TANH_TABLE.table.cuda()
        output = output.cuda()
      NndctTanhTableLookup(input,
                           Ttable,
                           output,
                           fragpos)
    else:
      output = super().forward(input)

    return output


@py_utils.register_quant_op
def Tanh(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Tanh(*args, **kwargs)
  return deephi_Tanh(*args, **kwargs)

