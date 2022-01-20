

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

import math
import torch
from torch.autograd import Variable

from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
from nndct_shared.utils import NndctOption
import pytorch_nndct.utils as py_utils
from .fix_ops import NndctScale

__all__ = ['AvgPool2d']


class deephi_AvgPool2d(torch.nn.modules.AvgPool2d):
  r"""DeePhi Conv2d operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_AvgPool2d, self).__init__(*args, **kwards)
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, input):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input])
    output = super().forward(input)

    # scale to DPU accuracy
    needScale = False
    scale = 1.0
    if self.node.node_attr(self.node.op.AttrName.KERNEL) == [3, 3]:
      needScale = True
      scale = 9.0 * 7.0 / 64.0
    elif self.node.node_attr(self.node.op.AttrName.KERNEL) == [5, 5]:
      needScale = True
      scale = 25.0 * 10.0 / 256.0
    elif self.node.node_attr(self.node.op.AttrName.KERNEL) in [[6, 6], [3, 6], [6, 3]]:
      needScale = True
      scale = 36.0 * 7.0 / 256.0
    elif self.node.node_attr(self.node.op.AttrName.KERNEL) == [7, 7]:
      needScale = True
      scale = 49.0 * 21.0 / 1024.0
    elif self.node.node_attr(self.node.op.AttrName.KERNEL) == [14, 14]:
      needScale = True
      scale = 196.0 * 21.0 / 4096.0
    else:
      rec = self.node.node_attr(self.node.op.AttrName.KERNEL)[0] * self.node.node_attr(self.node.op.AttrName.KERNEL)[1]
      max_factor =  math.ceil(math.log(rec * 128,2))
      diff = 1.0
      multi_factor = 0.0
      shift_factor = 0.0
      for shift_factor_ in range(max_factor):
        factor = round((2 ** shift_factor_)/rec)
        diff_ = abs(factor / (2 ** shift_factor_) - 1/rec)
        if diff_ < diff:
          multi_factor = factor
          diff = diff_
          shift_factor = shift_factor_
      scale = rec * multi_factor / (2 ** shift_factor)

    if needScale:
      NndctScale(output, scale)

    [output] = post_quant_process(self.node, [output])

    return output

@py_utils.register_quant_op
def AvgPool2d(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode is None or NndctOption.nndct_quant_off.value:
    return torch.nn.AvgPool2d(*args, **kwargs)
  return deephi_AvgPool2d(*args, **kwargs)
