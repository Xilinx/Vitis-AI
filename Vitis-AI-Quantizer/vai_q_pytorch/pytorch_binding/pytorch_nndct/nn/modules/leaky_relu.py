import torch
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
__all__ = ['LeakyReLU']


class deephi_LeakyReLU(torch.nn.LeakyReLU):
  r"""DeePhi LeakyReLU operation"""

  def __init__(self, *args, **kwargs):
    # only support the specified slope and inplace operation
    super().__init__(negative_slope=0.1015625, inplace=True)
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.valid_inputs = None
    self.valid_output = None
    self.params_name = None
    self.node = None
    self.need_quant_output = True

  def forward(self, tensors):
    inputs, _ = process_inputs_and_params(
        self.node,
        self.quant_mode,
        self.quantizer,
        inputs=tensors,
        valid_inputs=self.valid_inputs)
    output = super().forward(inputs)
    if (self.need_quant_output):
      [output] = post_quant_process(self.node, 
                                    self.valid_output, 
                                    [output],
                                    [output, output])
    return output


@py_utils.register_quant_op
def LeakyReLU(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode is None:
    return torch.nn.LeakyReLU(*args, **kwargs)
  return deephi_LeakyReLU(*args, **kwargs)
