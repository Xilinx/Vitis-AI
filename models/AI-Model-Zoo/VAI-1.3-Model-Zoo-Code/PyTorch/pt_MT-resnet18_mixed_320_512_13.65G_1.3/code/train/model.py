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
from data.config import solver
import os
import resnet as models

class MTNet(nn.Module):

    def __init__(self, num_classes, seg_classes):
        super(MTNet, self).__init__()
        self.num_classes = num_classes
        self.seg_classes = seg_classes
        resnet50_32s = models.resnet50(pretrained=True)
        
        resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion
        
        self.resnet50_32s = resnet50_32s
        
        self.score_2s = nn.Sequential(
            nn.Conv2d(64, self.seg_classes, kernel_size=1), 
            nn.BatchNorm2d(self.seg_classes))
        
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),)
        
        self.conv_block7 = nn.Sequential(
            nn.Conv2d(512, 128, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),)
        
        self.conv_block8 = nn.Sequential(
            nn.Conv2d(256, 128, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),)
        
        self.toplayer3 = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(1024))  # Reduce channels
        self.toplayer2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(512))
        self.toplayer1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(256))
        self.toplayer0 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64))       

        self.loc_0 = nn.Sequential(nn.Conv2d(256, 6 * 4, 3, padding=1), nn.BatchNorm2d(24))
        self.loc_1 = nn.Sequential(nn.Conv2d(512, 6 * 4, 3, padding=1), nn.BatchNorm2d(24))
        self.loc_2 = nn.Sequential(nn.Conv2d(1024, 6 * 4, 3, padding=1), nn.BatchNorm2d(24))
        self.loc_3 = nn.Sequential(nn.Conv2d(2048, 6 * 4, 3, padding=1), nn.BatchNorm2d(24))
        self.loc_4 = nn.Sequential(nn.Conv2d(512, 6 * 4, 3, padding=1), nn.BatchNorm2d(24))
        self.loc_5 = nn.Sequential(nn.Conv2d(256, 6 * 4, 3, padding=1), nn.BatchNorm2d(24))
        self.loc_6 = nn.Sequential(nn.Conv2d(256, 4 * 4, 3, padding=1), nn.BatchNorm2d(16))
        
        self.conf_0 = nn.Sequential(nn.Conv2d(256, 6 * self.num_classes, 3, padding=1), nn.BatchNorm2d(6*self.num_classes))
        self.conf_1 = nn.Sequential(nn.Conv2d(512, 6 * self.num_classes, 3, padding=1), nn.BatchNorm2d(6*self.num_classes))
        self.conf_2 = nn.Sequential(nn.Conv2d(1024, 6 * self.num_classes, 3, padding=1), nn.BatchNorm2d(6*self.num_classes))
        self.conf_3 = nn.Sequential(nn.Conv2d(2048, 6 * self.num_classes, 3, padding=1), nn.BatchNorm2d(6*self.num_classes))
        self.conf_4 = nn.Sequential(nn.Conv2d(512, 6 * self.num_classes, 3, padding=1), nn.BatchNorm2d(6*self.num_classes))
        self.conf_5 = nn.Sequential(nn.Conv2d(256, 6 * self.num_classes, 3, padding=1), nn.BatchNorm2d(6*self.num_classes))
        self.conf_6 = nn.Sequential(nn.Conv2d(256, 4 * self.num_classes, 3, padding=1), nn.BatchNorm2d(4*self.num_classes))
   
        self.priorbox = PriorBox(solver)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
 
    def forward(self, x):
        loc = list()
        conf = list()
        
        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)  
        x = self.resnet50_32s.relu(x)
        f_2s = x
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)
        f0 = x
        
        x = self.resnet50_32s.layer2(x)
        f1 = x
        
        x = self.resnet50_32s.layer3(x)
        f2 = x
        
        x = self.resnet50_32s.layer4(x)
        feature3 = x
        
        
        top3 = nn.functional.upsample(self.toplayer3(feature3), scale_factor=2)
        feature2 = top3 + f2
        top2 = nn.functional.upsample(self.toplayer2(feature2), scale_factor=2)
        feature1 = top2 + f1
        top1 = nn.functional.upsample(self.toplayer1(feature1), scale_factor=2)
        feature0 = top1 + f0
        
        seg_feature = nn.functional.upsample(self.toplayer0(feature0), scale_factor=2) + f_2s 
        logits_2s = self.score_2s(seg_feature)
         
        seg = nn.functional.upsample(logits_2s, scale_factor=2)
        
        
        
        feature4 = self.conv_block6(feature3)
        feature5 = self.conv_block7(feature4)
        feature6 = self.conv_block8(feature5)
        
        
        loc.append(self.loc_0(feature0).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_1(feature1).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_2(feature2).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_3(feature3).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_4(feature4).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_5(feature5).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_6(feature6).permute(0, 2, 3, 1).contiguous())
        
        conf.append(self.conf_0(feature0).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_1(feature1).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_2(feature2).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_3(feature3).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_4(feature4).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_5(feature5).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_6(feature6).permute(0, 2, 3, 1).contiguous())
        
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            seg,
            self.priors
        )
        return output
        
def build_model(det_classes, seg_classes):
    model = MTNet(det_classes, seg_classes)
    return model      
        

