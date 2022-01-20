#!/usr/bin/env bash

# Copyright 2021 Xilinx Inc.
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

import torch
import torch.nn as nn

class MyNet(nn.Module):
  def __init__(self):
    super(MyNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(32, 128, 3, stride=2)
    self.bn2 = nn.BatchNorm2d(128)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv3 = nn.Conv2d(128, 256, 3)
    self.bn3 = nn.BatchNorm2d(256)
    self.relu3 = nn.ReLU(inplace=True)
    self.conv4 = nn.Conv2d(256, 512, 3)
    self.bn4 = nn.BatchNorm2d(512)
    self.relu4 = nn.ReLU(inplace=True)
    self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
    self.fc1 = nn.Linear(512, 32)
    self.fc = nn.Linear(32, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu3(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = self.relu4(x)
    x = self.avgpool1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.fc(x)
    return x
