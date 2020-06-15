import torch
from nndct_shared.quantization import maybe_get_quantizer, process_inputs_and_params, post_quant_process
import pytorch_nndct.utils as py_utils
__all__ = ['Mean']

class deephi_Mean(torch.nn.Module):
  r"""DeePhi Concat operation"""

  def __init__(self, *args, **kwargs):
    super(deephi_Mean, self).__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    #self.__module_name = None
    self.valid_inputs, self.valid_output, self.quant_info, self.params_name = None, None, None, None
    self.node = None

  def forward(self, input, dim, keepdim):
    input, _ = process_inputs_and_params(
        self.node,
        self.quant_mode,
        self.quantizer,
        inputs=input,
        valid_inputs=self.valid_inputs)
    output = torch.mean(input, dim, keepdim)
    [output] = post_quant_process(self.node, self.valid_output, [output],
                                  [output, output])

    return output
  
  
@py_utils.register_quant_op
def Mean(*args, **kwargs):
  #quant_mode,_ = maybe_get_quantizer()
  #if quant_mode==None:
  #    return
  return deephi_Mean(*args, **kwargs)
