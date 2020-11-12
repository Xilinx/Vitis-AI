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

#-*-coding:utf-8-*-
import torch
import math
import numpy as np
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torchsummary import summary

class PointsQuality(nn.Module):
    def __init__(self):
        super(PointsQuality, self).__init__()
        channel = 3
        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU(inplace=True)
        # points branch
        # self.points_fc1 = nn.AdaptiveAvgPool2d(1)
        self.points_fc1 = nn.Linear(128 * 6 * 3, 128)
        # self.points_fc1 = nn.Linear(128 * 8 * 5, 128)
        self.points_relu5 = nn.ReLU(inplace=True)
        self.points_fc2 = nn.Linear(128, 10)
        # quality branch
        # self.quality_fc1 = nn.AdaptiveAvgPool2d(1)
        self.quality_fc1 = nn.Linear(128 * 6 * 3, 128)
        # self.quality_fc1 = nn.Linear(128 * 8 * 5, 128)
        self.quality_relu5 = nn.ReLU(inplace=True)
        # self.drop_q = nn.Dropout2d(p=0.5)
        self.quality_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = x.view(x.size(0), -1)
        # points branch
        x_p = self.points_fc1(x)
        x_p = self.points_relu5(x_p)
        x_p = self.points_fc2(x_p)
        # quality branch
        x_q = self.quality_fc1(x)
        x_q = self.quality_relu5(x_q)
        x_q = self.quality_fc2(x_q)
        return x_p, x_q

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.in_channels = 64
        self.layer1 = self.make_layer(block, 64, layers[0], 1)
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3], 2)
        self.layer5 = self.make_layer(block, 512, layers[4], 2)
        self.avg_pool = nn.AvgPool2d(kernel_size = (6, 5))
        self.fc_points = nn.Linear(512, 10)

    def make_layer(self, block, out_channels, blocks, stride = 1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(nn.Conv2d(in_channels = self.in_channels, out_channels = out_channels, 
                                                 kernel_size = 3, stride = stride, padding = 1),
                                       nn.BatchNorm2d(num_features = out_channels)
                                       )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out_points = self.fc_points(out)
        return out_points

def get_model():
    net = PointsQuality()
    # initialzation
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    return net

def get_model_resnet():
    net = ResNet(ResidualBlock, [2, 2, 2, 2, 2])
    # initialzation
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    return net
