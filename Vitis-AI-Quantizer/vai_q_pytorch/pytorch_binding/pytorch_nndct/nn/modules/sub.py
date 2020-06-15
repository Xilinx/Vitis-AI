import torch
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
__all__ = ['Sub']

class deephi_Sub(torch.nn.Module):

  def __init__(self):
    super(deephi_Sub, self).__init__()
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
    output = torch.sub(input=input, other=other, alpha=alpha)
    [output] = post_quant_process(self.node, 
                                  self.valid_output, 
                                  [output],
                                  [output, output])
    return output

@py_utils.register_quant_op
def Sub(*args, **kwargs):
  return deephi_Sub(*args, **kwargs)

