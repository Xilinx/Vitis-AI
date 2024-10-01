

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

import os
import torch
import torch.nn.functional as F  
from torch.autograd import Variable

class QuantStubF(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    return input.clone()

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output.clone()

class DeQuantStubF(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    return input.clone()

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output.clone()

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
 
class Correlation1DElemwiseF(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input_1:torch.Tensor, input_2:torch.Tensor, pad_size):
    output_dim =  pad_size + 1
    B, C, H, W = input_1.size()
    input_2 = F.pad(input_2, pad=(pad_size,0,0,0), mode="constant",value=0)
    cv = []
    for i in range(output_dim - 1, -1, -1):
        cost = input_1 * input_2[:, :, :, i:(i + W)]
        cost = cost.unsqueeze(2)
        cv.append(cost)
    cv = torch.cat(cv, 2)
    return cv
  
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output


class Correlation2DElemwiseF(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input_1:torch.Tensor, input_2:torch.Tensor, pad_size):
    output_dim = 2 * pad_size + 1
    B, C, H, W = input_1.size()
    input_2 = F.pad(input_2, [pad_size] * 4)
    cv = []
    for i in range(output_dim):
        for j in range(output_dim):
            cost = input_1 * input_2[:, :, i:(i + H), j:(j + W)]
            cost = cost.unsqueeze(2)
            cv.append(cost)
    cv = torch.cat(cv, 2)
    return cv
  
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output


class CostVolumeF(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input_1:torch.Tensor, input_2:torch.Tensor, maxdisp):
    if os.environ["DUMP_XMODEL"]=='1':
        cost = Variable(torch.zeros(input_1.size()[0], input_1.size()[1]*2, maxdisp//4,  input_1.size()[2],  input_1.size()[3])).cpu()
    else:
        cost = Variable(torch.zeros(input_1.size()[0], input_1.size()[1]*2, maxdisp//4,  input_1.size()[2],  input_1.size()[3])).cuda()
    
    for i in range(maxdisp//4):
        if i > 0 :
            cost[:, :input_1.size()[1], i, :,i:]   = input_1[:,:,:,i:]
            cost[:, input_1.size()[1]:, i, :,i:] = input_2[:,:,:,:-i]
        else:
            cost[:, :input_1.size()[1], i, :,:]   = input_1
            cost[:, input_1.size()[1]:, i, :,:]   = input_2
    output = cost.contiguous()
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output
