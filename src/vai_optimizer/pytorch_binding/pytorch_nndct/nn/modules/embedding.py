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
from nndct_shared.utils import NndctOption
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors 
import pytorch_nndct.utils as py_utils

__all__ = ['embedding']

class deephi_Embedding(torch.nn.modules.sparse.Embedding):
  r"""DeePhi transpose operation, support float and double"""

  def __init__(self, *args, **kwargs):
    super(deephi_Embedding, self).__init__(*args, **kwargs)
    self.params_name = None
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.param_quantized = False

  def forward(self, input):

    if self.quant_mode <= 0 or (not self.node.in_quant_part):
      return torch.nn.functional.embedding(input, self.weight, padding_idx=self.padding_idx)
      
    inplace = (NndctOption.nndct_quant_off.value or 
      self.quantizer is not None and self.quantizer.inplace)
    # params = []
    # qparams = []
    
    # param_names = []
    # for k in self.node.op.params.keys():
    #   pname = self.node.op.params[k].name
    #   p = getattr(self.quantizer.quant_model, k.value)
    #   param_names.append(pname)
    #   params.append(p)
    
    if not self.param_quantized:
      if inplace:
        _ = quantize_tensors([self.weight], 
                             self.node, 
                             tensor_names= [self.params_name[0]], 
                             tensor_type='param')
        qparams = [self.weight]
      else:
        qparams = quantize_tensors([self.weight], 
                                   self.node, 
                                   tensor_names= [self.params_name[0]], 
                                   tensor_type='param')
      if not NndctOption.nndct_quant_off.value:
        self.param_quantized = True
    else:
      qparams = [self.weight]
  
    inputs = quantize_tensors([input], self.node, tensor_type='input')
    output = torch.nn.functional.embedding(inputs[0], qparams[0], padding_idx=self.padding_idx)
    output = quantize_tensors([output], self.node)[0]
    
    return output

@py_utils.register_quant_op
def embedding(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Embedding(*args, **kwargs)
  return deephi_Embedding(*args, **kwargs)
