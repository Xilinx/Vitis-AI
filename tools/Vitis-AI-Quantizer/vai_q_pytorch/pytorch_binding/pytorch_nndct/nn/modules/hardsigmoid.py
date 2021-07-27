

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
import torch.nn.functional as F

from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
from nndct_shared.utils import NndctOption
import pytorch_nndct.utils as py_utils
from .fix_ops import NndctScale

__all__ = ['Hardsigmoid']


class deephi_Hardsigmoid(torch.nn.Module):
  r"""DeePhi Conv2d operation, support float and double"""

  def __init__(self, inplace=False, *args, **kwards):
    super(deephi_Hardsigmoid, self).__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None
    self.inplace = inplace

  def forward(self, input):
    if self.quant_mode is None or NndctOption.nndct_quant_off.value:
      return torch.div(F.relu6(torch.add(input, 3.)), 6.) 
    else:
      [input], _ = process_inputs_and_params(
          self.node,
          self.quantizer,
          inputs=[input])
      output = F.relu6(torch.add(input, 3.))

      # scale to DPU accuracy
      scale = 2731.0 / 16384.0
      NndctScale(output, scale)

      [output] = post_quant_process(self.node, [output])

      return output

@py_utils.register_quant_op
def Hardsigmoid(*args, **kwargs):
  return deephi_Hardsigmoid(*args, **kwargs)
