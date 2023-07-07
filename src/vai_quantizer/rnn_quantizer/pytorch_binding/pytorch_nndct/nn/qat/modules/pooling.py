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

from pytorch_nndct.nn.modules.fix_ops import NndctScale

class DPUAvgPool2d(torch.nn.modules.AvgPool2d):

  def forward(self, input):
    output = super().forward(input)

    # scale to DPU accuracy
    need_scale = False
    scale = 1.0
    if self.kernel_size == [3, 3]:
      need_scale = True
      scale = 9.0 * 7.0 / 64.0
    elif self.kernel_size == [5, 5]:
      need_scale = True
      scale = 25.0 * 10.0 / 256.0
    elif self.kernel_size == [[6, 6], [3, 6], [6, 3]]:
      need_scale = True
      scale = 36.0 * 7.0 / 256.0
    elif self.kernel_size == [7, 7]:
      need_scale = True
      scale = 49.0 * 21.0 / 1024.0
    elif self.kernel_size == [14, 14]:
      need_scale = True
      scale = 196.0 * 21.0 / 4096.0

    if need_scale:
      NndctScale(output, scale)

    return output

class DPUAdaptiveAvgPool2d(torch.nn.modules.AdaptiveAvgPool2d):

  def forward(self, input):
    output = super().forward(input)

    if (isinstance(self.output_size,
                   (tuple, list)) and tuple(self.output_size) !=
        (1, 1)) or self.output_size != 1:
      print(
          "[WARNING] For adaptive average pooling, DPU only supports output size=1"
      )


    need_scale = False
    scale = 1.0
    if input.shape[2] == 3 and input.shape[3] == 3:
      need_scale = True
      scale = 9.0 * 7.0 / 64.0
    elif input.shape[2] == 5 and input.shape[3] == 5:
      need_scale = True
      scale = 25.0 * 10.0 / 256.0
    elif input.shape[2] == 6 and input.shape[3] == 6:
      need_scale = True
      scale = 36.0 * 7.0 / 256.0
    elif input.shape[2] == 7 and input.shape[3] == 7:
      need_scale = True
      scale = 49.0 * 21.0 / 1024.0
    elif input.shape[2] == 14 and input.shape[3] == 14:
      need_scale = True
      scale = 196.0 * 21.0 / 4096.0

    if need_scale:
      NndctScale(output, scale)

    return output
