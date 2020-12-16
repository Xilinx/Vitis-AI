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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
from NormLinear import NormLinear
from Scale import Scale

__all__ = ['Resnet', 'resnet20']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            with torch.no_grad():
                m.bias.zero_()

class MyBlock(nn.Module):
    def __init__(self, planes, stride=1, downsample=None, nonlinear='relu'):
        super(MyBlock, self).__init__()
        self.downsample = downsample
        
        self.conv1 = conv3x3(planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        if nonlinear == 'relu':
            self.relu1 = nn.ReLU(inplace=False)
        else:
            self.relu1 = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if nonlinear == 'relu':
            self.relu2 = nn.ReLU(inplace=False)
        else:
            self.relu2 = nn.PReLU(planes)
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)

        return out

class MyNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, gray=False, nonlinear='relu', pruning=1):
        super(MyNet, self).__init__()
        input_dim = 1 if gray else 3
        self.nonlinear = nonlinear
        self.layer1 = self._make_layer(block, input_dim, 64//pruning, layers[0])
        self.layer2 = self._make_layer(block, 64//pruning, 128//pruning, layers[1])
        self.layer3 = self._make_layer(block, 128//pruning, 256//pruning, layers[2])
        self.layer4 = self._make_layer(block, 256//pruning, 512//pruning, layers[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        if self.nonlinear == 'relu':
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=False),
            )
        else:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=3, stride=2, padding=1, bias=True),
                nn.PReLU(planes),
            )
        '''
        nn.init.xavier_normal(downsample._modules['0'].weight)
        if downsample._modules['0'].bias is not None:
            nn.init.constant(downsample._modules['0'].bias, 0)
        '''

        layers = []
        layers.append(block(planes, stride, downsample, nonlinear=self.nonlinear))
        for i in range(1, blocks):
            layers.append(block(planes, nonlinear=self.nonlinear))

        return nn.Sequential(*layers)

class Resnet(nn.Module):
    def __init__(self,num_classes, layers=[1,2,4,1], pretrained=False, wn=True, fn=True, ring=None, sphere=False, gray=False, pruning=1, fea_dim=512, **kwargs):
        super(Resnet, self).__init__()
        self.fea_dim = fea_dim
        self.model = MyNet(MyBlock, layers, gray=gray, pruning=pruning, **kwargs)
        self.model.dropout = nn.Dropout2d(p=0.4)
        self.model.fc1 = nn.Linear(512//pruning*7*6, self.fea_dim, bias=False)

        #self.model.fc1_bn = nn.BatchNorm1d(self.fea_dim, affine=True)
        self.wn = wn
        self.fn = fn
        self.sphere = sphere
        self.model.classifier = NormLinear(self.fea_dim, num_classes, bias=False, 
                    wn=self.wn, fn=self.fn)

        if ring is not None:
            s = ring.get('s', 1)
            l = ring.get('l', False)
            rl = ring.get('rl', False)
            self.scale = Scale(s, l, rl)
        self.num_classes = num_classes
        # self.model.apply(weight_init)

    def forward(self, x):
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc1(x)
        if self.training:
            feature = x
            output = self.model.classifier(x)
            return feature, output
        else:
            return x


def resnet20(num_classes, wn=True, fn=True, ring=None, nonlinear='relu', fea_dim=512):
    layers = [1, 2, 4, 1]
    return Resnet(num_classes=num_classes, layers=layers, wn=wn, fn=fn, ring=ring, nonlinear=nonlinear, fea_dim=fea_dim)


