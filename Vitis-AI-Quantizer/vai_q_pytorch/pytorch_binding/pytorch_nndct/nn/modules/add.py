import torch
from nndct_shared.quantization.utils import maybe_get_quantizer
from nndct_shared.quantization.utils import post_quant_process
from nndct_shared.quantization.utils import process_inputs_and_params
import pytorch_nndct.utils as py_utils
__all__ = ['Add']

class deephi_Add(torch.nn.Module):

  def __init__(self):
    super(deephi_Add, self).__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.valid_inputs = None
    self.valid_output = None
    self.quant_info = None
    self.params_name = None
    self.node = None

  def forward(self, input, other, alpha=1):
    [input, other], _ = process_inputs_and_params(
        self.node,
        self.quant_mode,
        self.quantizer,
        inputs=[input, other],
        valid_inputs=self.valid_inputs)
    output = torch.add(input=input, other=other, alpha=alpha)
    [output] = post_quant_process(self.node, 
                                  self.valid_output, 
                                  [output],
                                  [output, output])
    return output
  
@py_utils.register_quant_op
def Add(*args, **kwargs):
  return deephi_Add(*args, **kwargs)
