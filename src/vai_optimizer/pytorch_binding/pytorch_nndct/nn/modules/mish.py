

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

__all__ = ['Mish']

class deephi_Mish(torch.nn.Mish):
  r"""DeePhi Mish operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_Mish, self).__init__(*args, **kwards)
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()

  def forward(self, input):
    # quantize parameters
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    output = F.mish(qinput)
    output = quantize_tensors([output], self.node)[0]
    return output

@py_utils.register_quant_op
def Mish(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Mish(*args, **kwargs)
  return deephi_Mish(*args, **kwargs)
