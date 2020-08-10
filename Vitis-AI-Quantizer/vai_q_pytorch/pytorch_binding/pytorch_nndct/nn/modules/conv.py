import torch
from torch.autograd import Variable

from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
__all__ = ['Conv2d']

class deephi_Conv2d(torch.nn.modules.conv.Conv2d):
  r"""DeePhi Conv2d operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_Conv2d, self).__init__(*args, **kwards)
    self.valid_inputs = None
    self.valid_output = None
    self.params_name = None
    self.bias_valid_inputs = None
    self.bias_valid_output = None
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.param_quantized = False
    self.need_quant_output = True

  def forward(self, input):
    if self.bias is not None:
      params = [self.weight, self.bias]
    else:
      params = [self.weight]

    [input], __ = process_inputs_and_params(
        self.node,
        self.quant_mode,
        self.quantizer,
        inputs=[input],
        valid_inputs=self.valid_inputs,
        params=[],
        param_names=[])
    if (not self.param_quantized):
      __, __ = process_inputs_and_params(
          self.node,
          self.quant_mode,
          self.quantizer,
          inputs=[],
          valid_inputs=[],
          params=params,
          param_names=self.params_name)
      self.param_quantized = True

    output = super().forward(input)
    if (self.need_quant_output):
      [output] = post_quant_process(self.node, self.valid_output, [output],
                                    [output, output])

    return output
  
@py_utils.register_quant_op
def Conv2d(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Conv2d(*args, **kwargs)
  return deephi_Conv2d(*args, **kwargs)
