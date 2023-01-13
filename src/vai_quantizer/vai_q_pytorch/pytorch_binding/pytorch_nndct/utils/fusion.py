# Copyright 2022 Xilinx Inc.
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

def fuse_conv_bn(conv, bn, inplace=True):
  assert (not (conv.training or
               bn.training)), "Fusion only for evaluation mode!"
  fused_conv = conv if inplace else copy.deepcopy(conv)

  fused_conv.weight, fused_conv.bias = fuse_conv_bn_weight(
      fused_conv.weight, fused_conv.bias, bn.running_mean, bn.running_var,
      bn.weight, bn.bias, bn.eps, conv.transposed)
  return fused_conv

def fuse_conv_bn_weight(conv_weight,
                        conv_bias,
                        running_mean,
                        running_var,
                        gamma,
                        beta,
                        eps,
                        transposed=False):
  if conv_bias is None:
    conv_bias = torch.zeros_like(running_mean)
  if gamma is None:
    gamma = torch.ones_like(running_mean)
  if beta is None:
    beta = torch.zeros_like(running_mean)
  var_rsqrt = torch.rsqrt(running_var + eps)

  if transposed:
    shape = [1, -1] + [1] * (len(conv_weight.shape) - 2)
  else:
    shape = [-1, 1] + [1] * (len(conv_weight.shape) - 2)

  conv_weight = conv_weight * (gamma * var_rsqrt).reshape(shape)
  conv_bias = (conv_bias - running_mean) * var_rsqrt * gamma + beta

  return torch.nn.Parameter(conv_weight), torch.nn.Parameter(conv_bias)
