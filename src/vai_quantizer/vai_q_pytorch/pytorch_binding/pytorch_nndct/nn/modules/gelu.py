

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
from distutils.version import LooseVersion

__all__ = ['GELU']

class deephi_GELU(torch.nn.GELU):
  r"""DeePhi GELU operation, support float and double"""

  def __init__(self, approximate=None):
    if torch.__version__ >= LooseVersion('1.12.0'):
      super(deephi_GELU, self).__init__(approximate)
    else:
      super(deephi_GELU, self).__init__()
      self.approximate = approximate 
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()

  def gelu(self, x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

  def generate_gelu_table(self, bw, input_scale, output_scale, table_name):
    def gelu_numpy(x):
      """Gaussian Error Linear Unit.

      This is a smoother version of the RELU.
      Original paper: https://arxiv.org/abs/1606.08415
      Args:
        x: float Tensor to perform activation.

      Returns:
        `x` with the GELU activation applied.
      """
      cdf = 0.5 * (1.0 + np.tanh(
          (np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
      return x * cdf

    inputs_table_positive = np.arange(0, 2**(bw - 1)).astype(np.float32)
    inputs_table_negatice = np.arange(-(2**(bw - 1)), 0).astype(np.float32)
    inputs_table = np.hstack((inputs_table_positive, inputs_table_negatice))

    outputs_table = gelu_numpy(inputs_table / (2**(input_scale)))
    outputs_table_int = np.clip(np.round(outputs_table * (2**(output_scale))), -(2**(bw - 1)), 2**(bw - 1) - 1)

    return outputs_table_int

  def forward(self, input):
    if self.quant_mode <= 0 or (not self.node.in_quant_part):
      return super().forward(input)

    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]

    if NndctOption.nndct_quant_off.value or self.quantizer.exporting:
      output = super().forward(qinput)

    else:
      # Method 1: Dynamic table look up with 8 bw
      if NndctOption.nndct_op_gelu_mode.value == "dynamic_table" or NndctOption.nndct_ip_v70_bert.value:
        output = self.gelu(qinput)
        output = quantize_tensors([output], self.node)[0]

        # # Code for Golden Verification
        # input_name = self.node.in_nodes[0]
        # input_node = self.quantizer.configer.get_Nndctnode(input_name)
        # if not self.quantizer.configer.node_output_quantizable(input_node):
        #   input_name = input_node.in_nodes[0]
        # bw = self.quantizer.get_quant_config(self.node.name, False)[0]
        # fragpos = self.quantizer.get_quant_config(input_name, False)[1]
        # output_scale = self.quantizer.get_quant_config(self.node.name, False)[1]
        # gelu_table = self.generate_gelu_table(bw, fragpos, output_scale, self.node.name)
        # qinput_int = torch.where(qinput >=0, qinput * (2**(fragpos)), qinput * (2**(fragpos)) + 2**bw)
        # gelu_table = torch.from_numpy(gelu_table).to(input.device)
        # output = gelu_table[qinput_int.to(torch.long)]
        # output = output / (2**(output_scale))

      # Method 0: Quant input and output of Softmax  
      elif torch.__version__ >= LooseVersion('1.12.0'):
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
