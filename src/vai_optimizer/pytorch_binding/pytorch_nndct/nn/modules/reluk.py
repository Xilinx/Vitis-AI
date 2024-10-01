

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
from .function import RelukF

class ReLUk(torch.nn.Module):
    def __init__(self, channel_max=6.0):
        super(ReLUk, self).__init__()
        self.channel_max = channel_max

    def forward(self, input):
        return RelukF.apply(input, self.channel_max)
'''
class ReLU_max(torch.nn.Module):
    def __init__(self, channel_max):
        super(ReLU_max, self).__init__()
        self.func = ReLUk()
        self.channel_max = channel_max
    
    def forward(self, input):
        input = self.func(input, self.channel_max)
        return input
'''