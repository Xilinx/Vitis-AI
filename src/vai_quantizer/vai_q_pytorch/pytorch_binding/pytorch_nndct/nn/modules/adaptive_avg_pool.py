

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

import pytorch_nndct.utils as py_utils
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors 
from nndct_shared.utils import NndctOption, NndctScreenLogger

__all__ = ['AdaptiveAvgPool2d']


class deephi_AdaptiveAvgPool2d(torch.nn.modules.AdaptiveAvgPool2d):
  r"""DeePhi Conv2d operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_AdaptiveAvgPool2d, self).__init__(*args, **kwards)
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()

  def forward(self, input):
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    output = super().forward(qinput)

    input_size = [int(dim) for dim in input.shape[2:]]
    mod = [input_size[i] % self.output_size[i] for i in range(0, len(input_size))]
    if mod != [0] * len(mod):
      if self.node is not None:
        NndctScreenLogger().warning_once(f"AdaptiveAvgpool2d op({self.node.name}) is not quantized. Because it's output size {self.output_size} are not factor of input size {input_size}.")
      return output
    # During slow trace, the dim of shape will convert to tensor value which is not support in nndct.
    kernel = [int(input_size[i] / self.output_size[i]) for i in range(0, len(input_size))]
    # scale to DPU accuracy
    if NndctOption.nndct_avg_pool_approximate.value:
      scale = 1.0
      if kernel == [3, 3]:
        scale = 9.0 * 7.0 / 64.0
      elif kernel == [5, 5]:
        scale = 25.0 * 10.0 / 256.0
      elif kernel in [[6, 6], [3, 6], [6, 3]]:
        scale = 36.0 * 7.0 / 256.0
      elif kernel == [7, 7]:
        scale = 49.0 * 21.0 / 1024.0
      elif kernel == [14, 14]:
        scale = 196.0 * 21.0 / 4096.0
      else:
        rec = kernel[0] * kernel[1]
        max_factor = math.ceil(math.log(rec * 128, 2))
        diff = 1.0
        multi_factor = 0.0
        shift_factor = 0.0
        for shift_factor_ in range(max_factor):
          factor = round((2 ** shift_factor_)/rec)
          diff_ = abs(factor / (2 ** shift_factor_) - 1 / rec)
          if diff_ < diff:
            multi_factor = factor
            diff = diff_
            shift_factor = shift_factor_
        scale = rec * multi_factor / (2 ** shift_factor)

      output = output * scale

    output = quantize_tensors([output], self.node)[0]

    return output
  
@py_utils.register_quant_op
def AdaptiveAvgPool2d(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode is None or NndctOption.nndct_quant_off.value:
    return torch.nn.AdaptiveAvgPool2d(*args, **kwargs)
  return deephi_AdaptiveAvgPool2d(*args, **kwargs)
