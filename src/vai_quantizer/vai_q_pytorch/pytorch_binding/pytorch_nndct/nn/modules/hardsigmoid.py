

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
from nndct_shared.quantization import quantize_tensors
from nndct_shared.utils import NndctOption
import pytorch_nndct.utils as py_utils

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
      qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
      output = F.relu6(torch.add(qinput, 3.))

      # scale to DPU accuracy
      scale = 2731.0 / 16384.0
      output = output * scale

      output = quantize_tensors([output], self.node)[0]

      return output

@py_utils.register_quant_op
def Hardsigmoid(*args, **kwargs):
  return deephi_Hardsigmoid(*args, **kwargs)
