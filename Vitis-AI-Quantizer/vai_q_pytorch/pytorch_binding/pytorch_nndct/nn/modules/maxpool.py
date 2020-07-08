import torch
from torch.autograd import Variable

from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
__all__ = ['MaxPool2d']

class deephi_MaxPool2d(torch.nn.modules.MaxPool2d):
  r"""DeePhi Conv2d operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_MaxPool2d, self).__init__(*args, **kwards)
    self.valid_inputs = None
    self.valid_output = None
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.need_quant_output = True

  def forward(self, input):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quant_mode,
        self.quantizer,
        inputs=[input],
        valid_inputs=self.valid_inputs)
    output = super().forward(input)
    if (self.need_quant_output):
      [output] = post_quant_process(self.node, self.valid_output, [output],
                                    [output, output])
    return output
  
@py_utils.register_quant_op
def MaxPool2d(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.MaxPool2d(*args, **kwargs)
  return deephi_MaxPool2d(*args, **kwargs)
