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

from .resnet import ResNet

class Baseline(nn.Module):

    def __init__(self, num_classes=10, last_stride=1, in_planes = 2048):
        super().__init__()
        self.in_planes = in_planes
        self.base = ResNet(last_stride)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.num_classes = num_classes
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.norm = nn.functional.normalize
        self.scale = 14

    def forward(self, x):
        global_feat = self.gap(self.base(x))  
        global_feat = global_feat.view(global_feat.shape[0], -1)  
        feat = self.bottleneck(global_feat)
        return feat
        #return self.norm(feat)

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

