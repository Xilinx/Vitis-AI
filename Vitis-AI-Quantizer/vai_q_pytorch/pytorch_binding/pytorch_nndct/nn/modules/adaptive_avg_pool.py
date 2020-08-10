import torch

import pytorch_nndct.utils as py_utils
from nndct_shared.quantization import (maybe_get_quantizer, post_quant_process,
                                       process_inputs_and_params)
from nndct_shared.utils import NndctOption

from .fix_ops import NndctScale

__all__ = ['AdaptiveAvgPool2d']


class deephi_AdaptiveAvgPool2d(torch.nn.modules.AdaptiveAvgPool2d):
  r"""DeePhi Conv2d operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_AdaptiveAvgPool2d, self).__init__(*args, **kwards)
    self.valid_inputs = None
    self.valid_output = None
    self.params_name = None
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()

  def forward(self, input):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quant_mode,
        self.quantizer,
        inputs=[input],
        valid_inputs=self.valid_inputs)
    output = super().forward(input)

    
    # scale to DPU accuracy
    if self.output_size != [1, 1]:
      print(
          "NNDCT-Waring: For adaptive average pooling, DPU only supports output size 1"
      )
    needScale = False
    scale = 1.0
    if input.shape[2] == 3 and input.shape[3] == 3:
      needScale = True
      scale = 9.0 * 7.0 / 64.0
    elif input.shape[2] == 5 and input.shape[3] == 5:
      needScale = True
      scale = 25.0 * 10.0 / 256.0
    elif input.shape[2] == 6 and input.shape[3] == 6:
      needScale = True
      scale = 36.0 * 7.0 / 256.0
    elif input.shape[2] == 7 and input.shape[3] == 7:
      needScale = True
      scale = 49.0 * 21.0 / 1024.0
    elif input.shape[2] == 14 and input.shape[3] == 14:
      needScale = True
      scale = 196.0 * 21.0 / 4096.0

    if needScale:
      NndctScale(output, scale)

    [output] = post_quant_process(self.node, self.valid_output, [output],
                                  [output, output])

    return output
  
@py_utils.register_quant_op
def AdaptiveAvgPool2d(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode is None or NndctOption.nndct_quant_off.value:
    return torch.nn.AdaptiveAvgPool2d(*args, **kwargs)
  return deephi_AdaptiveAvgPool2d(*args, **kwargs)
