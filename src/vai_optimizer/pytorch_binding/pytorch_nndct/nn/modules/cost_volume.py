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
from .function import CostVolumeF


class Cost_Volume(torch.nn.Module):
  def __init__(self, maxdisp=1):
    super(Cost_Volume, self).__init__()
    self.maxdisp = maxdisp
  
  def forward(self, input_1, input_2):
    return CostVolumeF.apply(input_1, input_2, self.maxdisp)
