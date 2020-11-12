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


# MIT License

# Copyright (c) 2019 Hengshuang Zhao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
from code.models.mobilenet import mobilenet_v2
from code.models import resnet

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['BaseNet']


class BaseNet(nn.Module):
    def __init__(self, backbone, pretrained=False, dilated=False, multi_grid=False, deep_base=False, norm_layer=nn.BatchNorm2d):
        super(BaseNet, self).__init__()
        self.deep_base = deep_base
        self.backbone = backbone

        if backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=pretrained, dilated=dilated, multi_grid=multi_grid,
                                              deep_base=deep_base, norm_layer=norm_layer)
        elif backbone.lower() == 'mobilenetv2':
            self.pretrained = mobilenet_v2(pretrained=pretrained)
        else:
            print('Only support resnet18 and mobilenetv2 as backbone!')
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        if self.backbone.lower() == 'mobilenetv2':
            c=[]
            for i, (j,m) in enumerate(self.pretrained.features.named_children()):
                x = m(x)
                if i in [3, 6, 13, 18]:  # 24, 32, 96, 1280
                    c.append(x)
            return c
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

