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

# PART OF THIS FILE AT ALL TIMES.

import math

import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Module

__all__ = ['NormLinear']

class NormLinear(Module):
    def __init__(self, in_features, out_features, bias=False, wn=True, fn=True):
        super(NormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wn = wn
        self.fn = fn
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight_norm = self.weight
        feature_norm = input
        if self.wn:
            weight_norm = F.normalize(self.weight, p=2, dim=1)
        if self.fn:
            feature_norm = F.normalize(input, p=2, dim=1)
        return self.linear(feature_norm, weight_norm)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', wn=' + str(self.wn) \
            + ', fn=' + str(self.fn) + ')'

    def linear(self, input, weight):
        """
        Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
        Shape:
            - Input: :math:`(N, *, in\_features)` where `*` means any number of
              additional dimensions
            - Weight: :math:`(out\_features, in\_features)`
            - Bias: :math:`(out\_features)`
            - Output: :math:`(N, *, out\_features)`
        """
        if input.dim() == 2 and self.bias is not None:
            # fused op is marginally faster
            return torch.addmm(self.bias, input, weight.t())
    
        output = input.matmul(weight.t())
        if self.bias is not None:
            output += self.bias
        return output

if __name__ == '__main__':
    model = NormLinear(4, 1, wn=False, fn=False)
    x = Variable(torch.Tensor([[1, 2, 3, 4]]), requires_grad=True)
    y = model(x)
    print('x: {} y: {}'.format(x, y))
    y.backward()
    print('x.grad: {}'.format(x.grad))
    print('weight.grad: {}'.format(model.weight.grad))

