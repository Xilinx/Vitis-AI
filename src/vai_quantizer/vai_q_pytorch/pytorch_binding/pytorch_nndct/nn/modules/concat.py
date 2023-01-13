

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
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors
import pytorch_nndct.utils as py_utils
__all__ = ['Cat']

class deephi_Cat(torch.nn.Module):
  r"""DeePhi Concat operation"""

  def __init__(self, *args, **kwargs):
    super(deephi_Cat, self).__init__()
    # self.dim = kwargs.get('dim', 0)
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, tensors, dim):
    qinputs = quantize_tensors(tensors, self.node, tensor_type='input')
    output = torch.cat(qinputs, dim)
    output = quantize_tensors([output], self.node)[0]

    return output
  
@py_utils.register_quant_op
def Cat(*args, **kwargs):
  return deephi_Cat(*args, **kwargs)
