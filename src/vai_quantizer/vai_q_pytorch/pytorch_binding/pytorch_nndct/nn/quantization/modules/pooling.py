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
import math

# from pytorch_nndct.nn.modules.fix_ops import NndctScale

class DPUAvgPool2d(torch.nn.modules.AvgPool2d):

  def forward(self, input):
    output = super().forward(input)

    # scale to DPU accuracy
    scale = 1.0
    if self.kernel_size == [3, 3]:
      scale = 9.0 * 7.0 / 64.0
    elif self.kernel_size == [5, 5]:
      scale = 25.0 * 10.0 / 256.0
    elif self.kernel_size in [[6, 6], [3, 6], [6, 3]]:
      scale = 36.0 * 7.0 / 256.0
    elif self.kernel_size == [7, 7]:
      scale = 49.0 * 21.0 / 1024.0
    elif self.kernel_size == [14, 14]:
      scale = 196.0 * 21.0 / 4096.0
    else:
      rec = self.kernel_size[0] * self.kernel_size[1]
      max_factor = math.ceil(math.log(rec * 128, 2))
      diff = 1.0
      multi_factor = 0.0
      shift_factor = 0.0
      for shift_factor_ in range(max_factor):
        factor = round((2**shift_factor_) / rec)
        diff_ = abs(factor / (2**shift_factor_) - 1 / rec)
        if diff_ < diff:
          multi_factor = factor
          diff = diff_
          shift_factor = shift_factor_
      scale = rec * multi_factor / (2**shift_factor)

    # NndctScale(output, scale)
    output = output * scale

    return output

class DPUAdaptiveAvgPool2d(torch.nn.modules.AdaptiveAvgPool2d):

  def forward(self, input):
    output = super().forward(input)

    if (isinstance(self.output_size,
                   (tuple, list)) and tuple(self.output_size) !=
        (1, 1)) or self.output_size != 1:
      print(
          "[WARNING] For AdaptiveAvgPooling, DPU only supports output size=1"
      )

    scale = 1.0
    if input.shape[2] == 3 and input.shape[3] == 3:
      scale = 9.0 * 7.0 / 64.0
    elif input.shape[2] == 5 and input.shape[3] == 5:
      scale = 25.0 * 10.0 / 256.0
    elif (input.shape[2] == 6 and input.shape[3] == 6) or (
        input.shape[2] == 3 and input.shape[3] == 6) or (input.shape[2] == 6 and
                                                         input.shape[3] == 3):
      scale = 36.0 * 7.0 / 256.0
    elif input.shape[2] == 7 and input.shape[3] == 7:
      scale = 49.0 * 21.0 / 1024.0
    elif input.shape[2] == 14 and input.shape[3] == 14:
      scale = 196.0 * 21.0 / 4096.0
    else:
      rec = input.shape[2] * input.shape[3]
      max_factor = math.ceil(math.log(rec * 128, 2))
      diff = 1.0
      multi_factor = 0.0
      shift_factor = 0.0
      for shift_factor_ in range(max_factor):
        factor = round((2**shift_factor_) / rec)
        diff_ = abs(factor / (2**shift_factor_) - 1 / rec)
        if diff_ < diff:
          multi_factor = factor
          diff = diff_
          shift_factor = shift_factor_
      scale = rec * multi_factor / (2**shift_factor)

    # NndctScale(output, scale)
    output = output * scale

    return output
