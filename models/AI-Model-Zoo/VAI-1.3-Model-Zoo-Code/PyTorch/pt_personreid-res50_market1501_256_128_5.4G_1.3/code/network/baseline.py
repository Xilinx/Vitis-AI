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


# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
import torch.nn.functional as F

from .resnet import ResNet, BasicBlock, Bottleneck

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, model_name, num_classes=10, last_stride=1):
        super().__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                                block=BasicBlock,
                                layers=[2, 2, 2, 2])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                                block=Bottleneck,
                                layers=[3, 4, 6, 3])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.num_classes = num_classes
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.norm = nn.functional.normalize

    def forward(self, x):
        feat = self.base(x)
        global_feat = self.gap(feat)  
        global_feat = global_feat.view(-1, self.in_planes)  
        feat = self.bottleneck(global_feat)
        return feat
        #return self.norm(feat)

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

