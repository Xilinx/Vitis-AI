

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
from nndct_shared.nndct_graph import Tensor
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
from pytorch_nndct.utils import TorchOpClassType


__all__ = ['Module']


def creat_module(torch_op_type, torch_op_attr, *args, **kwargs):
  if torch_op_attr.op_class_type == TorchOpClassType.NN_MODULE:
    # creat module for module
    module_cls = getattr(torch.nn, torch_op_type, None)
    if module_cls:
      class deephi_Module(module_cls):
        r"""quantizable operation"""

        def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.node = None
          self.quant_mode, self.quantizer = maybe_get_quantizer()
        
        def extra_repr(self):
          return f"'{module_cls.__name__}'"
        
        def forward(self, *inputs, **kwargs):
          inputs, _ = process_inputs_and_params(
              self.node,
              self.quantizer,
              inputs=list(inputs))
          
          output = super().forward(*inputs, **kwargs)
          [output] = post_quant_process(self.node, [output])
          # output = super().forward(*inputs, **kwargs)

          return output
      
      return deephi_Module(*args, **kwargs)
  
  elif torch_op_attr.op_class_type in [TorchOpClassType.NN_FUNCTION, TorchOpClassType.TORCH_FUNCTION]:
    # create module for function
    if getattr(torch.nn.functional, torch_op_type, None):
      caller = getattr(torch.nn.functional, torch_op_type)
    else:
      caller = getattr(torch, torch_op_type, None)
      
    if caller:
      class deephi_Func_Module(torch.nn.Module):
        r"""quantizable operation"""

        def __init__(self, caller, *args, **kwards):
          super().__init__()
          self.node = None
          self.quant_mode, self.quantizer = maybe_get_quantizer()
          self.caller = caller
        
        def extra_repr(self):
          return f"'{caller.__name__}'"
         
        def forward(self, *args, **kwargs):
             
          inputs = []
          
          def collect_inputs(inputs, value):
            if isinstance(value, torch.Tensor):
              inputs.append(value)
            elif isinstance(value, (tuple, list)):
              for i in value:
                collect_inputs(inputs, i)
                
          for k, v in kwargs.items():     
            collect_inputs(inputs, v)
          
          inptus, _ = process_inputs_and_params(
              self.node,
              self.quantizer,
              inputs=inputs)
                  
          output = caller(*args, **kwargs)
          
          [output] = post_quant_process(self.node, [output])
            
          return output
        
      return deephi_Func_Module(caller, *args, **kwargs) 
    
  elif torch_op_attr.op_class_type == TorchOpClassType.TENSOR:
     # create module for method
    if getattr(torch.Tensor, torch_op_type, None):
      class deephi_Tensor_Module(torch.nn.Module):
        r"""quantizable operation"""

        def __init__(self, op_type, *args, **kwards):
          super().__init__()
          self.node = None
          self.quant_mode, self.quantizer = maybe_get_quantizer()
          self.op_type = op_type
        
        def extra_repr(self):
          return f"'{self.op_type}'"
        
        def forward(self, input, *args, **kwargs):
          [input], _ = process_inputs_and_params(
              self.node,
              self.quantizer,
              inputs=[input])
          
          output = getattr(input, self.op_type, None)(*args, **kwargs)
          
          [output] = post_quant_process(self.node, [output])

          return output 
          
      return deephi_Tensor_Module(torch_op_type, *args, **kwargs) 
   
  else:
    raise RuntimeError("Unkown op type:{torch_op_type}")
   
      
def Module(nndct_type, *args, **kwargs):
  torch_op_type = py_utils.get_torch_op_type(nndct_type)
  torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)
  return creat_module(torch_op_type, torch_op_attr, *args, **kwargs)
