

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
import torch.nn.functional as F  

class QuantStubF(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    return input

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output

class DeQuantStubF(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    return input

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output

class RelukF(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input:torch.Tensor, channel_max):
    #_size = input.size()
    #channelwise_clamp_value = channel_max.repeat(repeats=(_size[0], 1, _size[2], _size[3]))
    #channelwise_clamp_value =  channel_max.expand(_size[0], 1, _size[2], _size[3])
    #channel_max_tensor = torch.Tensor(channel_max).to(input.device)
    input = F.relu(input) - F.relu(input-channel_max)
    return input
  
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output
  
class ChannelScaleF(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input:torch.Tensor, channel_scale):
    input = input * channel_scale
    return input
  
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output
 
