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
from torch.autograd import Variable
from layers import *
import os
from collections import OrderedDict
import resnet as models

solver = {
    'k1': 8,
    'k2': 8,
    'act_clip_val': 8,
    'warmup': False,
    'det_classes': 4,
    'seg_classes': 16,
    'lr_steps': (12000, 18000),
    #'lr_steps': (5, 10),
    'max_iter': 20010,
    'feature_maps': [(80,128), (40,64), (20,32), (10,16), (5,8), (3,6), (1,4)],
    'resize': (320,512),
    'steps': [4, 8, 16, 32, 64, 128, 256],
    'min_sizes': [10, 30, 60, 100, 160, 220, 280],
    'max_sizes': [30, 60, 100, 160, 220, 280, 340],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
    'variance': [0.1, 0.2],
    'clip': False,
}
class MTNet(nn.Module):

    def __init__(self, num_classes, seg_classes):
        super(MTNet, self).__init__()
        self.num_classes = num_classes
        self.seg_classes = seg_classes
        resnet18_32s = models.resnet18(pretrained=False)
        resnet_block_expansion_rate = resnet18_32s.layer1[0].expansion
        
        self.resnet18_32s = resnet18_32s
        
        self.score_2s = nn.Sequential(OrderedDict([
            ('score_2s_conv',nn.Conv2d(64, self.seg_classes, kernel_size=1)), 
            ('score_2s_BN',nn.BatchNorm2d(self.seg_classes)),]))
        
        self.conv_block6 = nn.Sequential(OrderedDict([
            ('conv_block6_conv1',nn.Conv2d(512, 256, 1, padding=0)),
            ('conv_block6_BN1',nn.BatchNorm2d(256)),
            ('conv_block6_relu1',nn.ReLU(inplace=True)),
            ('conv_block6_conv2',nn.Conv2d(256, 512, 3, padding=1, stride=2)),
            ('conv_block6_BN2',nn.BatchNorm2d(512)),
            ('conv_block6_relu2',nn.ReLU(inplace=True)),]))
        
        self.conv_block7 = nn.Sequential(OrderedDict([
            ('conv_block7_conv1',nn.Conv2d(512, 128, 1, padding=0)),
            ('conv_block7_BN1',nn.BatchNorm2d(128)),
            ('conv_block7_relu1',nn.ReLU(inplace=True)),
            ('conv_block7_conv2',nn.Conv2d(128, 256, 3, padding=0)),
            ('conv_block7_BN2',nn.BatchNorm2d(256)),
            ('conv_block7_relu2',nn.ReLU(inplace=True)),]))
        
        self.conv_block8 = nn.Sequential(OrderedDict([
            ('conv_block8_conv1',nn.Conv2d(256, 128, 1, padding=0)),
            ('conv_block8_BN1',nn.BatchNorm2d(128)),
            ('conv_block8_relu1',nn.ReLU(inplace=True)),
            ('conv_block8_conv2',nn.Conv2d(128, 256, 3, padding=0)),
            ('conv_block8_BN2',nn.BatchNorm2d(256)),
            ('conv_block8_relu2',nn.ReLU(inplace=True)),]))
        
        self.toplayer3 = nn.Sequential(OrderedDict([
            ('toplayer3_conv',nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)),
            ('toplayer3_BN',nn.BatchNorm2d(256)),]))  # Reduce channels
        self.toplayer2 = nn.Sequential(OrderedDict([
            ('toplayer2_conv',nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)),
            ('toplayer2_BN',nn.BatchNorm2d(128)),]))  # Reduce channels
        self.toplayer1 = nn.Sequential(OrderedDict([
            ('toplayer1_conv',nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)),
            ('toplayer1_BN',nn.BatchNorm2d(64)),]))  # Reduce channels
        self.toplayer0 = nn.Sequential(OrderedDict([
            ('toplayer0_conv',nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)),
            ('toplayer0_BN',nn.BatchNorm2d(64)),]))  # Reduce channels

        self.loc_0 = nn.Sequential(OrderedDict([
            ('loc_0_conv',nn.Conv2d(64, 6 * 6, 3, padding=1)),
            ('loc_0_BN', nn.BatchNorm2d(36)),]))
        self.loc_1 = nn.Sequential(OrderedDict([
            ('loc_1_conv',nn.Conv2d(128, 6 * 6, 3, padding=1)),
            ('loc_1_BN', nn.BatchNorm2d(36)),]))
        self.loc_2 = nn.Sequential(OrderedDict([
            ('loc_2_conv',nn.Conv2d(256, 6 * 6, 3, padding=1)),
            ('loc_2_BN', nn.BatchNorm2d(36)),]))
        self.loc_3 = nn.Sequential(OrderedDict([
            ('loc_3_conv',nn.Conv2d(512, 6 * 6, 3, padding=1)),
            ('loc_3_BN', nn.BatchNorm2d(36)),]))
        self.loc_4 = nn.Sequential(OrderedDict([
            ('loc_4_conv',nn.Conv2d(512, 6 * 6, 3, padding=1)),
            ('loc_4_BN', nn.BatchNorm2d(36)),]))
        self.loc_5 = nn.Sequential(OrderedDict([
            ('loc_5_conv',nn.Conv2d(256, 6 * 6, 3, padding=1)),
            ('loc_5_BN', nn.BatchNorm2d(36)),]))
        self.loc_6 = nn.Sequential(OrderedDict([
            ('loc_6_conv',nn.Conv2d(256, 4 * 6, 3, padding=1)),
            ('loc_6_BN', nn.BatchNorm2d(24)),]))

        self.conf_0 = nn.Sequential(OrderedDict([
             ('conf_0_conv',nn.Conv2d(64, 6 * self.num_classes, 3, padding=1)),
             ('conf_0_BN',nn.BatchNorm2d(6 * self.num_classes)),
             ]))
        self.conf_1 = nn.Sequential(OrderedDict([
             ('conf_1_conv',nn.Conv2d(128, 6 * self.num_classes, 3, padding=1)),
             ('conf_1_BN',nn.BatchNorm2d(6 * self.num_classes)),
             ]))

        self.conf_2 = nn.Sequential(OrderedDict([
             ('conf_2_conv',nn.Conv2d(256, 6 * self.num_classes, 3, padding=1)),
             ('conf_2_BN',nn.BatchNorm2d(6 * self.num_classes)),
             ]))
        self.conf_3 = nn.Sequential(OrderedDict([
             ('conf_3_conv',nn.Conv2d(512, 6 * self.num_classes, 3, padding=1)),
             ('conf_3_BN',nn.BatchNorm2d(6 * self.num_classes)),
             ]))
        self.conf_4 = nn.Sequential(OrderedDict([
             ('conf_4_conv',nn.Conv2d(512, 6 * self.num_classes, 3, padding=1)),
             ('conf_4_BN',nn.BatchNorm2d(6 * self.num_classes)),
             ]))
        self.conf_5 = nn.Sequential(OrderedDict([
             ('conf_5_conv',nn.Conv2d(256, 6 * self.num_classes, 3, padding=1)),
             ('conf_5_BN',nn.BatchNorm2d(6 * self.num_classes)),
             ]))
        self.conf_6 = nn.Sequential(OrderedDict([
             ('conf_6_conv',nn.Conv2d(256, 4 * self.num_classes, 3, padding=1)),
             ('conf_6_BN',nn.BatchNorm2d(4 * self.num_classes)),
             ]))
   
        #self.priorbox = PriorBox(solver)
        #self.priors = Variable(self.priorbox.forward(), volatile=True)
 
    def forward(self, x):
        loc = list()
        conf = list()
        
        x = self.resnet18_32s.conv1(x)
        x = self.resnet18_32s.bn1(x)  
        x = self.resnet18_32s.relu(x)
        f_2s = x
        x = self.resnet18_32s.maxpool(x)

        x = self.resnet18_32s.layer1(x)
        f0 = x
        
        x = self.resnet18_32s.layer2(x)
        f1 = x
        
        x = self.resnet18_32s.layer3(x)
        f2 = x
        
        x = self.resnet18_32s.layer4(x)
        feature3 = x
        
        
        top3 = nn.functional.upsample(self.toplayer3(feature3), scale_factor=2,mode='bilinear')
        feature2 = top3 + f2
        top2 = nn.functional.upsample(self.toplayer2(feature2), scale_factor=2,mode='bilinear')
        feature1 = top2 + f1
        top1 = nn.functional.upsample(self.toplayer1(feature1), scale_factor=2,mode='bilinear')
        feature0 = top1 + f0
        
        seg_feature = nn.functional.upsample(self.toplayer0(feature0), scale_factor=2,mode='bilinear') + f_2s 
        logits_2s = self.score_2s(seg_feature)
         
        seg = nn.functional.upsample(logits_2s, scale_factor=2,mode='bilinear')
        
        
        
        feature4 = self.conv_block6(feature3)
        feature5 = self.conv_block7(feature4)
        feature6 = self.conv_block8(feature5)
        
        
        loc.append(self.loc_0(feature0))#.permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_1(feature1))#.permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_2(feature2))#.permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_3(feature3))#.permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_4(feature4))#.permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_5(feature5))#.permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_6(feature6))#.permute(0, 2, 3, 1).contiguous())
        
        conf.append(self.conf_0(feature0))#.permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_1(feature1))#.permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_2(feature2))#.permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_3(feature3))#.permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_4(feature4))#.permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_5(feature5))#.permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_6(feature6))#.permute(0, 2, 3, 1).contiguous())
        
        #loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        #conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        output = (
            loc,#.view(loc.size(0), -1, 6),
            conf,#.view(conf.size(0), -1, self.num_classes),
            seg,
            #self.priors
        )
        return output
        
def build_model(det_classes, seg_classes):
    model = MTNet(det_classes, seg_classes)
    return model      
        
