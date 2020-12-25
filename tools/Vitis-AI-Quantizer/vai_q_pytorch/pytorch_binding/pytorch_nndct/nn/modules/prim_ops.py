

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
import torch.nn.functional as F 
from nndct_shared.base import NNDCT_CONSTANT
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
from nndct_shared.quantization import quant_reluk_params
import pytorch_nndct.utils as py_utils
from typing import Any, Optional, Sequence, Union
# __all__ = ["Int", "strided_slice", "Input", "slice_tensor_inplace_copy"]


class _PrimModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.params_name = None
    self.node = None
    
  def forward(*args, **kwargs):
    pass
  

class deephi_Int(_PrimModule):
  
  def __init__(self):
    super().__init__()

  def forward(self, input):
    output = int(input)
    return output 
  
  
@py_utils.register_quant_op
def Int(*args, **kwargs):
  return deephi_Int(*args, **kwargs)


class deephi_QuantInput(_PrimModule):
  def __init__(self):
    super().__init__()
    
  def forward(self, input):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input],
    )

    output = input

    [output] = post_quant_process(self.node, [output])  
    return output

@py_utils.register_quant_op  
def quant_input(*args, **kwargs):
  return deephi_QuantInput(*args, **kwargs)
  
class deephi_Input(_PrimModule):
  
  def __init__(self):
    super().__init__()
  
  def forward(self, input):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input],
    )
    # check input shape
    if self.node.out_tensors[0].is_complete_tensor() and self.node.out_tensors[0].ndim == 4:
      py_utils.blob_to_torch_format(self.node.out_tensors[0])
      if not (self.node.out_tensors[0].shape[1:] == list(input.size())[1:]):
        raise RuntimeError(f"The shape of input ({input.size()}) should be the same with that of dummy input ({[None] + self.node.out_tensors[0].shape[1:]})")
      py_utils.blob_to_nndct_format(self.node.out_tensors[0])
    output = input

    if self.node.in_quant_part:
      [output] = post_quant_process(self.node, [output])

    return output
  
  
@py_utils.register_quant_op  
def Input(*args, **kwargs):
  return deephi_Input(*args, **kwargs)


class deephi_StridedSlice(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, start, end, step):
    size = input.size()
    for i in range(len(start)):
      if end[i] == NNDCT_CONSTANT.INT_MAX:
        end[i] = size[i]
      indices = torch.arange(start[i], end[i], step[i]).to(input.device)
      input = torch.index_select(input, i, indices)
    
    output = input
    
    return output 
  
  
@py_utils.register_quant_op 
def strided_slice(*args, **kwargs):
  return deephi_StridedSlice(*args, **kwargs)


class deephi_SliceInplaceCopy(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, source, dim, index):
    index = torch.tensor([index]).to(input.device)
    output = input.index_copy_(dim, index, source.unsqueeze(dim))
    return output 
  
  
@py_utils.register_quant_op 
def slice_tensor_inplace_copy(*args, **kwargs):
  return deephi_SliceInplaceCopy(*args, **kwargs)
      
        
class deephi_Index(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, index):
    if len(index) == 2:
      if index[0] is None:
        output = input[:, index[1]]
      elif index[1] is None:
        output = input[index[0], :]
      else:
        output = input[index[0], index[1]]
    else:
      output = input[index]
    return output 
  
  
@py_utils.register_quant_op 
def Index(*args, **kwargs):
  return deephi_Index(*args, **kwargs)    


class deephi_IndexInputInplace(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, indices, values, accumulate):
    # TODO: try to remove hard code 
    
    if any([len(index.tolist()) == 0 for index in indices if index is not None]):
      return input
    
    if indices[0] is None:
      input[:, indices[1]] = values
    elif indices[1] is None and len(indices) == 2:
      input[indices[0], :] = values
    elif all([index is not None for index in indices]):
      input[indices] = values
      
    return input 
  
  
@py_utils.register_quant_op 
def index_put_inplace(*args, **kwargs):
  return deephi_IndexInputInplace(*args, **kwargs)


class deephi_ReLUk(_PrimModule):
  def __init__(self):
    super().__init__()
  
  def forward(self, input:torch.Tensor, channel_max:Union[torch.Tensor, Sequence[Any], float]):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input],
    )
    
    if isinstance(channel_max, (list, tuple)):
      channel_max = torch.Tensor(channel_max).to(input.device)
    elif isinstance(channel_max, float):
      channel_max = torch.Tensor([channel_max]).to(input.device)
    if self.node.in_quant_part:
      channel_max = quant_reluk_params(self.node, channel_max)
    
    output = F.relu(input) - F.relu(input-channel_max)
    
    if self.node.in_quant_part:
      [output] = post_quant_process(self.node, [output])
    
    return output

@py_utils.register_quant_op 
def Reluk(*args, **kwargs):
  return deephi_ReLUk(*args, **kwargs)
