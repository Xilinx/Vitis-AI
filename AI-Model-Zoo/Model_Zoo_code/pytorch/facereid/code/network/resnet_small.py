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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch import nn
import torchvision.models as models
from .resnet_model import resnet18
from ipdb import set_trace

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



class Resnetsmall(nn.Module):

    def __init__(self, num_classes=10, last_stride=1, in_planes = 128):
        super(Resnetsmall, self).__init__()
        self.in_planes = in_planes
        self.base = resnet18(channels=[24,32,64,128,256])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.num_classes = num_classes
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        self.norm = nn.functional.normalize
        self.scale = 14

    def forward(self, x):
        
        global_feat = self.gap(self.base(x))  
        global_feat = global_feat.view(global_feat.shape[0], -1)  
        feat = self.bottleneck(global_feat)
        if self.training:
            normed_feat = self.norm(feat)
            normed_weight = self.norm(self.classifier.weight)
            cosine = self.scale * normed_feat.matmul(normed_weight.t())
            return cosine, normed_feat
        else:
            return feat
            #return self.norm(feat)

