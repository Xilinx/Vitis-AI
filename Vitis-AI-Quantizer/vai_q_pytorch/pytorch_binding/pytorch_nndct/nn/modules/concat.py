import torch
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
__all__ = ['Cat']

class deephi_Cat(torch.nn.Module):
  r"""DeePhi Concat operation"""

  def __init__(self, *args, **kwargs):
    super(deephi_Cat, self).__init__()
    # self.dim = kwargs.get('dim', 0)
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.valid_inputs = None
    self.valid_output = None
    self.params_name = None
    self.node = None
    self.need_quant_output = True

  def forward(self, tensors, dim):
    inputs, _ = process_inputs_and_params(
        self.node,
        self.quant_mode,
        self.quantizer,
        inputs=tensors,
        valid_inputs=self.valid_inputs)
    output = torch.cat(inputs, dim)
    if (self.need_quant_output):
      [output] = post_quant_process(self.node, self.valid_output, [output],
                                    [output, output])

    return output
  
@py_utils.register_quant_op
def Cat(*args, **kwargs):
  #quant_mode,_ = maybe_get_quantizer()
  #if quant_mode==None:
  #    return
  return deephi_Cat(*args, **kwargs)
