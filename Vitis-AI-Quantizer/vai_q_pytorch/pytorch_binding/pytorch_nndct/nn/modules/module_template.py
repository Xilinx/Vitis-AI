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
          self.valid_inputs = None
          self.valid_output = None
          self.node = None
          self.quant_mode, self.quantizer = maybe_get_quantizer()
          self.need_quant_output = True
        
        def extra_repr(self):
          return f"'{module_cls.__name__}'"
        
        def forward(self, *inputs, **kwargs):
          if self.quantizer and self.quantizer.configer.is_node_quantizable(self.node, lstm=False):
            inputs, _ = process_inputs_and_params(
                self.node,
                self.quant_mode,
                self.quantizer,
                inputs=list(inputs),
                valid_inputs=self.valid_inputs)
            
            output = super().forward(*inputs, **kwargs)
            if (self.need_quant_output):
              [output] = post_quant_process(self.node, self.valid_output, [output],
                                            [output, output])
          else:
            output = super().forward(*inputs, **kwargs)

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
          self.valid_inputs = None
          self.valid_output = None
          self.node = None
          self.quant_mode, self.quantizer = maybe_get_quantizer()
          self.need_quant_output = True
          self.caller = caller
          self._match_inputs = []
        
        def extra_repr(self):
          return f"'{caller.__name__}'"
         
        def forward(self, *args, **kwargs):
          if len(self._match_inputs) == 0:
            def _check_kwargs(value):
              if isinstance(value, Tensor) and value in self.node.in_tensors:
                return True
              elif isinstance(value, (tuple, list)):
                check_result = [_check_kwargs(i) for i in value]
                return any(check_result)              
              
            for key in kwargs.keys():
              if _check_kwargs(self.node.node_config(key)):
                self._match_inputs.append(key)
                
          for key in self._match_inputs:
            if isinstance(kwargs[key], (tuple, list)):
              inputs = kwargs[key]
            else:
              inputs = [kwargs[key]]
             
          if self.quantizer and self.quantizer.configer.is_node_quantizable(self.node, lstm=False):
            inptus, _ = process_inputs_and_params(
                self.node,
                self.quant_mode,
                self.quantizer,
                inputs=inputs,
                valid_inputs=self.valid_inputs)
            
            if isinstance(kwargs[key], (tuple, list)):
              kwargs[key] = inputs
            else:
              kwargs[key] = inputs[0]
          
            output = caller(*args, **kwargs)
          
            if (self.need_quant_output):
              [output] = post_quant_process(self.node, self.valid_output, [output],
                                            [output, output])
          else:
            output = caller(*args, **kwargs)
            
          return output
        
      return deephi_Func_Module(caller, *args, **kwargs) 
    
  elif torch_op_attr.op_class_type == TorchOpClassType.TENSOR:
     # create module for method
    if getattr(torch.Tensor, torch_op_type, None):
      class deephi_Tensor_Module(torch.nn.Module):
        r"""quantizable operation"""

        def __init__(self, op_type, *args, **kwards):
          super().__init__()
          self.valid_inputs = None
          self.valid_output = None
          self.node = None
          self.quant_mode, self.quantizer = maybe_get_quantizer()
          self.need_quant_output = True
          self.op_type = op_type
        
        def extra_repr(self):
          return f"'{self.op_type}'"
        
        def forward(self, input, *args, **kwargs):
          if self.quantizer and self.quantizer.configer.is_node_quantizable(self.node, lstm=False):
            [input], _ = process_inputs_and_params(
                self.node,
                self.quant_mode,
                self.quantizer,
                inputs=[input],
                valid_inputs=self.valid_inputs)
            
            output = getattr(input, self.op_type, None)(*args, **kwargs)
            
            if (self.need_quant_output):
              [output] = post_quant_process(self.node, self.valid_output, [output],
                                            [output, output])
          else:
            output = getattr(input, self.op_type, None)(*args, **kwargs)

          return output 
          
      return deephi_Tensor_Module(torch_op_type, *args, **kwargs) 
   
  else:
    raise RuntimeError("Unkown op type:{torch_op_type}")
   
      
def Module(nndct_type, *args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  torch_op_type = py_utils.get_torch_op_type(nndct_type)
  torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)
  return creat_module(torch_op_type, torch_op_attr, *args, **kwargs)
