

#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()

  def forward(self, input):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input])
    output = super().forward(input)

    # scale to DPU accuracy
    if (isinstance(self.output_size, (tuple, list)) and tuple(self.output_size) != (1, 1)) or self.output_size != 1:
      print(
          "NNDCT-Waring: For adaptive average pooling, DPU only supports output size 1"
      )
      
    kernel = [input.shape[3], input.shape[2]]
    self.node.set_node_attr(self.node.op.AttrName.KERNEL, kernel)
    self.node.set_node_attr(self.node.op.AttrName.STRIDE, kernel)   
    
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

    [output] = post_quant_process(self.node, [output])

    return output
  
@py_utils.register_quant_op
def AdaptiveAvgPool2d(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode is None or NndctOption.nndct_quant_off.value:
    return torch.nn.AdaptiveAvgPool2d(*args, **kwargs)
  return deephi_AdaptiveAvgPool2d(*args, **kwargs)
