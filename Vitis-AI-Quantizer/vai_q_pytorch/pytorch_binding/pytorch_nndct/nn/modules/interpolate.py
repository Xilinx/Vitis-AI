import torch

from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
__all__ = ['Interpolate']

class deephi_Interpolate(torch.nn.Module):

  def __init__(self, *args, **kwards):
    super(deephi_Interpolate, self).__init__(*args, **kwards)
    self.valid_inputs = None
    self.valid_output = None
    self.params_name = None
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()

  def forward(self,
              input,
              size=None,
              scale_factor=None,
              mode='nearest',
              align_corners=None):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quant_mode,
        self.quantizer,
        inputs=[input],
        valid_inputs=self.valid_inputs,
    )

    output = torch.nn.functional.interpolate(input, size, scale_factor, mode,
                                             align_corners)

    [output] = post_quant_process(self.node, self.valid_output, [output],
                                  [output, output])

    return output
  
@py_utils.register_quant_op
def Interpolate(*args, **kwargs):
  return deephi_Interpolate(*args, **kwargs)
