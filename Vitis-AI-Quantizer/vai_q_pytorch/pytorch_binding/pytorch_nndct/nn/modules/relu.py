import torch
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
__all__ = ['ReLU']

class deephi_ReLU(torch.nn.ReLU):
  r"""DeePhi ReLU operation"""

  def __init__(self, *args, **kwargs):
    super(deephi_ReLU, self).__init__(*args, **kwargs)
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.valid_inputs = None
    self.valid_output = None
    self.params_name = None
    self.node = None
    self.need_quant_output = False

  def forward(self, inputs):
    inputs, _ = process_inputs_and_params(
    self.node,
    self.quant_mode,
    self.quantizer,
    inputs = inputs,
    valid_inputs = self.valid_inputs)
    output = super().forward(inputs)
    if (self.need_quant_output):
      [output] = post_quant_process(self.node, self.valid_output, [output],
                                    [output, output])

    return output
  
@py_utils.register_quant_op
def ReLU(*args, **kwargs):
  #quant_mode,_ = maybe_get_quantizer()
  #if quant_mode==None:
  #    return
  return deephi_ReLU(*args, **kwargs)
