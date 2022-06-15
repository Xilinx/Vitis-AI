

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
#:
import torch
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors
from nndct_shared.utils import NndctOption
import pytorch_nndct.utils as py_utils
from nndct_shared.utils import calculate_op_scale
__all__ = ['Mean']

class deephi_Mean(torch.nn.Module):
  r"""DeePhi Concat operation"""

  def __init__(self, *args, **kwargs):
    super(deephi_Mean, self).__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, input, dim, keepdim):
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    output = torch.mean(qinput, dim, keepdim)
    if len(self.node.node_attr(self.node.op.AttrName.DIMS)) == 1:
      scale = calculate_op_scale(self.node.in_tensors[0].shape[self.node.node_attr(self.node.op.AttrName.DIMS)[0]], self.node)
      output = output * scale

    output = quantize_tensors([output], self.node)[0]

    return output
  
  
@py_utils.register_quant_op
def Mean(*args, **kwargs):
  #quant_mode,_ = maybe_get_quantizer()
  #if quant_mode==None:
  #    return
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode is None or NndctOption.nndct_quant_off.value:
    return torch.mean
  return deephi_Mean(*args, **kwargs)
