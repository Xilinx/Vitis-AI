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
import copy

import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Module

__all__ = ['Scale']

class Scale(Module):
    def __init__(self, scale, learnable=False, ringlearn=False):
        super(Scale, self).__init__()
        self.learnable = learnable
        self.ringlearn = ringlearn
        if not self.learnable:
            self.scale = Variable(torch.Tensor([scale]).cuda())
        else:
            self.scale = Parameter(torch.Tensor([scale]).cuda())

    def forward(self, input):
        if self.ringlearn:
            notlearn_scale = copy.copy(self.scale)
            return notlearn_scale * input
        else:
            return self.scale * input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'learnable=' + str(self.learnable) \
            + ', ringlearn=' + str(self.ringlearn) + ')'


if __name__ == '__main__':
    model = Scale(30, learnable=True, ringlearn=False)
    x = Variable(torch.Tensor([[1, 2, 3, 4]]), requires_grad=True)
    y = model(x)
    y = torch.sum(y)
    print('x: {} y: {}'.format(x, y))
    y.backward()
    print('x.grad: {}'.format(x.grad))
    print('scale.grad: {}'.format(model.scale.grad))
    print(model)

