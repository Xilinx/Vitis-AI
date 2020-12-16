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


from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import interpolate
from code.models.resnet import BasicBlock, conv1x1, conv3x3
from code.models.base import BaseNet


class SemiBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SemiBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class UpsampleModule(nn.Module):
    def __init__(self, in_chs, decoder_chs, norm_layer):
        super(UpsampleModule, self).__init__()

        self.down_conv = conv1x1(in_chs, decoder_chs)
        self.down_bn = norm_layer(decoder_chs)
        downsample = nn.Sequential(
            self.down_conv,
            self.down_bn,
        )
        self.conv_enc = BasicBlock(in_chs, decoder_chs, downsample=downsample, norm_layer=norm_layer)
        self.conv_out = SemiBasicBlock(decoder_chs, decoder_chs, norm_layer=norm_layer)
        self.conv_up = nn.ConvTranspose2d(decoder_chs, decoder_chs, kernel_size=2, stride=2, bias=False)

    def forward(self, enc, prev):
        enc = self.conv_up(prev) + self.conv_enc(enc)
        dec = self.conv_out(enc)
        return dec

class FPN(BaseNet):
    def __init__(self, nclass, backbone, pretrained=False, norm_layer=nn.BatchNorm2d):
        super(FPN, self).__init__(backbone=backbone)
        in_chs_dict = {"mobilenetv2": 1280, "resnet18": 512}
        in_chs = in_chs_dict[backbone]
        self.head = FPNHead(in_chs, nclass, norm_layer, backbone)

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)
        
        outputs = []
        x = self.head(c1, c2, c3, c4)
        outputs.append(x)
        return tuple(outputs)


class FPNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, backbone):
        super(FPNHead, self).__init__()
        decoder_chs = out_channels
        if backbone == 'resnet18':
            layer_chs_list = [512, 256, 128, 64]
        elif backbone == 'mobilenetv2':
            layer_chs_list = [1280, 96, 32, 24]
        else:
            raise ValueError
        self.conv_enc2dec = conv3x3(layer_chs_list[0], decoder_chs)
        self.bn_enc2dec = norm_layer(out_channels)
        self.relu_enc2dec = nn.ReLU(True)

        self.up3 = UpsampleModule(layer_chs_list[1], decoder_chs, norm_layer)
        self.up2 = UpsampleModule(layer_chs_list[2], decoder_chs, norm_layer)
        self.up1 = UpsampleModule(layer_chs_list[3], decoder_chs, norm_layer)

        self.conv_up0 = nn.ConvTranspose2d(decoder_chs, decoder_chs, kernel_size=2, stride=2, bias=False)
        self.conv_up1 = nn.ConvTranspose2d(decoder_chs, out_channels, kernel_size=2, stride=2, bias=False)

        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                for p in m.parameters():
                    init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                for p in m.parameters():
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, c1, c2, c3, c4):
        c4 = self.relu_enc2dec(self.bn_enc2dec(self.conv_enc2dec(c4)))
        #print(c1.shape, c2.shape, c3.shape, c4.shape)
        c3 = self.up3(c3, c4)
        c2 = self.up2(c2, c3)
        c1 = self.up1(c1, c2)

        c1 = self.conv_up0(c1)
        c1 = self.conv_up1(c1)
        return c1


def get_fpn(nclass=19, backbone='resnet18', pretrained=False):
    assert backbone.lower() in ['resnet18', 'mobilenetv2'], NotImplementedError
    model = FPN(nclass=nclass, backbone=backbone, pretrained=pretrained)
    return model
