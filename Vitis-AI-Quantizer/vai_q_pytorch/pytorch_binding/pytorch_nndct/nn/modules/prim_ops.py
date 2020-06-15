import torch
from nndct_shared.base import NNDCT_CONSTANT
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils

# __all__ = ["Int", "strided_slice", "Input", "slice_tensor_inplace_copy"]


class _PrimModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.valid_inputs = None
    self.valid_output = None
    self.quant_info = None
    self.params_name = None
    self.node = None
    self.need_quant_output = None
    
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


class deephi_Input(_PrimModule):
  
  def __init__(self):
    super().__init__()
  
  def forward(self, input):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quant_mode,
        self.quantizer,
        inputs=[input],
        valid_inputs=self.valid_inputs,
    )
    # check input shape
    if self.node.out_tensors[0].is_complete_tensor() and self.node.out_tensors[0].ndim == 4:
      py_utils.blob_to_torch_format(self.node.out_tensors[0])
      if not (self.node.out_tensors[0].shape[1:] == list(input.size())[1:]):
        raise RuntimeError(f"The shape of input ({input.size()}) should be the same with that of dummy input ({[None] + self.node.out_tensors[0].shape[1:]})")
      py_utils.blob_to_nndct_format(self.node.out_tensors[0])
    output = input

    [output] = post_quant_process(self.node, self.valid_output, [output],
                                  [output, output])  
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
      indices = torch.arange(start[i], end[i], step[i]).cuda()
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
    index = torch.tensor([index]).cuda()
    output = input.index_copy_(dim, index, source.unsqueeze(dim))
    return output 
  
  
@py_utils.register_quant_op 
def slice_tensor_inplace_copy(*args, **kwargs):
  return deephi_SliceInplaceCopy(*args, **kwargs)
      
        
class deephi_Index(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, index):
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
