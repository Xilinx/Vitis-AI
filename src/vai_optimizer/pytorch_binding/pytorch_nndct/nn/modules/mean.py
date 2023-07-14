

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
    if self.quantizer is None or self.quant_mode is None or NndctOption.nndct_quant_off.value:
      return self._fp32_forward(input, dim, keepdim)
    else:
      return self._forward(input, dim, keepdim)

  def _fp32_forward(self, input, dim, keepdim):
    if dim is None:
      output = torch.mean(input)
    else:
      output = torch.mean(input, dim, keepdim)

    return output

  def _forward(self, input, dim, keepdim):
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    if dim is None:
      output = torch.mean(qinput)
    else:
      output = torch.mean(qinput, dim, keepdim)

    input_shape = self.node.in_tensors[0].shape
    if self.node.node_attr(self.node.op.AttrName.DIMS) == [None]:
      dim_list = [i for i in range(len(input_shape))]
    else:
      dim_list = self.node.node_attr(self.node.op.AttrName.DIMS)

    rec = 1
    for i in dim_list:
      input_shape = input.shape[i].item() if isinstance(input.shape[i], torch.Tensor) else input.shape[i]
      rec = rec * input_shape

    if (rec & (rec - 1)) != 0:
      scale = calculate_op_scale(rec, self.node)
      output = output * scale

    output = quantize_tensors([output], self.node)[0]

    return output
  
  
@py_utils.register_quant_op
def Mean(*args, **kwargs):
  return deephi_Mean(*args, **kwargs)
