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

# Copyright (c) 2018 davidtvs

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

import torch.nn as nn
import torch
from torch.autograd import Variable

__all__ = ['ENet', 'InitialBlock', 'RegularBottleneck', 'DownsamplingBottleneck', 'UpsamplingBottleneck']

class InitialBlock(nn.Module):
    """The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.
    """

    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1, padding=1,bias=False):
        super(InitialBlock,self).__init__()

        # the extension branch
        self.main_conv = nn.Conv2d(in_channels,out_channels - 3,kernel_size=kernel_size,stride=2,padding=1,bias=False)
        self.main_bn = nn.BatchNorm2d(out_channels - 3)
        self.main_relu = nn.ReLU(True)

        # Extension branch
        self.ext_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # main
        main = self.main_conv(x)
        main = self.main_bn(main)
        main = self.main_relu(main)
        # ext
        ext = self.ext_pool(x)
        # Concatenate branches
        out = torch.cat((main, ext), 1)
        return out


class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.

    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    """

    def __init__(self,
                 channels,internal_ratio=4,kernel_size=3,padding=0,dilation=1):
        super(RegularBottleneck,self).__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        # 1x1 projection convolution
        self.ext1_conv = nn.Conv2d(channels,internal_channels,kernel_size=1,stride=1,bias=False)
        self.ext1_bn = nn.BatchNorm2d(internal_channels)
        self.ext1_relu = nn.ReLU(True)

        self.ext2_conv = nn.Conv2d(internal_channels, internal_channels, \
                                   kernel_size=kernel_size, stride=1,padding=padding,dilation=dilation,bias=False)
        self.ext2_bn = nn.BatchNorm2d(internal_channels)
        self.ext2_relu = nn.ReLU(True)

        # 1x1 expansion convolution
        self.ext3_conv = nn.Conv2d(internal_channels,channels,kernel_size=1,stride=1,bias=False)
        self.ext3_bn = nn.BatchNorm2d(channels)
        self.ext3_relu = nn.ReLU(True)

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext1 = self.ext1_conv(x)
        ext1 = self.ext1_bn(ext1)
        ext1 = self.ext1_relu(ext1)

        ext2 = self.ext2_conv(ext1)
        ext2 = self.ext2_bn(ext2)
        ext2 = self.ext2_relu(ext2)

        ext3 = self.ext3_conv(ext2)
        ext3 = self.ext3_bn(ext3)
        ext3 = self.ext3_relu(ext3)
        # Add main and extension branches
        out = main + ext3
        return out


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.

    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.

    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    """

    def __init__(self, in_channels,out_channels,internal_ratio=4,kernel_size=3,padding=0):
        super(DownsamplingBottleneck,self).__init__()
        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        self.main_conv = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False)
        self.main_bn = nn.BatchNorm2d(out_channels)
        self.main_relu = nn.ReLU(True)

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext1_conv = nn.Conv2d(in_channels,internal_channels,kernel_size=3,stride=2,padding=1,bias=False)
        self.ext1_bn = nn.BatchNorm2d(internal_channels)
        self.ext1_relu = nn.ReLU(True)

        self.ext2_conv = nn.Conv2d(internal_channels, internal_channels, \
                                   kernel_size=kernel_size,stride=1,padding=padding,bias=False)
        self.ext2_bn = nn.BatchNorm2d(internal_channels)
        self.ext2_relu = nn.ReLU(True)

        self.ext3_conv = nn.Conv2d(internal_channels,out_channels,kernel_size=1,stride=1,bias=False)
        self.ext3_bn = nn.BatchNorm2d(out_channels)
        self.ext3_relu = nn.ReLU(True)

    def forward(self, x):
        # Main branch shortcut
        main = self.main_conv(x)
        main = self.main_bn(main)
        main = self.main_relu(main)
        main = self.main_max(main)

        # Extension branch
        ext1 = self.ext1_conv(x)
        ext1 = self.ext1_bn(ext1)
        ext1 = self.ext1_relu(ext1)

        ext2 = self.ext2_conv(ext1)
        ext2 = self.ext2_bn(ext2)
        ext2 = self.ext2_relu(ext2)

        ext3 = self.ext3_conv(ext2)
        ext3 = self.ext3_bn(ext3)
        ext3 = self.ext3_relu(ext3)
        # Add main and extension branches
        out = main + ext3

        return out


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.

    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.

    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.
    """

    def __init__(self,in_channels,out_channels,internal_ratio=4,kernel_size=3,padding=0):
        super(UpsamplingBottleneck,self).__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = int(in_channels // internal_ratio)

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.main_bn = nn.BatchNorm2d(out_channels)
        self.main_relu = nn.ReLU(True)
        # upsample
        self.main_up = nn.Upsample(scale_factor=2, mode='nearest')

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext1_conv = nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False)
        self.ext1_bn = nn.BatchNorm2d(internal_channels)
        self.ext1_relu = nn.ReLU(True)

        # Transposed convolution
        self.ext2_conv = nn.ConvTranspose2d(internal_channels,internal_channels,kernel_size=2,stride=2,padding=0,bias=False)
        self.ext2_bn = nn.BatchNorm2d(internal_channels)
        self.ext2_relu = nn.ReLU(True)

        # 1x1 expansion convolution
        self.ext3_conv = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.ext3_bn = nn.BatchNorm2d(out_channels)
        self.ext3_relu = nn.ReLU(True)

    def forward(self, x):
        # Main branch shortcut
        main = self.main_conv(x)
        main = self.main_bn(main)
        main = self.main_relu(main)
        main = self.main_up(main)
        # Extension branch
        ext1 = self.ext1_conv(x)
        ext1 = self.ext1_bn(ext1)
        ext1 = self.ext1_relu(ext1)

        ext2 = self.ext2_conv(ext1)
        ext2 = self.ext2_bn(ext2)
        ext2 = self.ext2_relu(ext2)

        ext3 = self.ext3_conv(ext2)
        ext3 = self.ext3_bn(ext3)
        ext3 = self.ext3_relu(ext3)

        # Add main and extension branches
        out = main + ext3
        return out


class ENet(nn.Module):
    """Generate the ENet model.

    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.

    """

    def __init__(self, num_classes):
        super(ENet,self).__init__()

        self.initial_block = InitialBlock(3, 16, padding=1)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(16,64,padding=1)
        self.regular1_1 = RegularBottleneck(64, padding=1)
        self.regular1_2 = RegularBottleneck(64, padding=1)
        self.regular1_3 = RegularBottleneck(64, padding=1)
        self.regular1_4 = RegularBottleneck(64, padding=1)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(64,128,padding=1)
        self.regular2_1 = RegularBottleneck(128, padding=1)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2)
        self.asymmetric2_3 = RegularBottleneck(128,kernel_size=5,padding=2)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4)
        self.regular2_5 = RegularBottleneck(128, padding=1)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8)
        self.asymmetric2_7 = RegularBottleneck(128,kernel_size=5,padding=2)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(128, padding=1)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2)
        self.asymmetric3_2 = RegularBottleneck(128,kernel_size=5,padding=2)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4)
        self.regular3_4 = RegularBottleneck(128, padding=1)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8)
        self.asymmetric3_6 = RegularBottleneck(128,kernel_size=5,padding=2)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16)
   
        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(128, 64, padding=1)
        self.regular4_1 = RegularBottleneck(64, padding=1)
        self.regular4_2 = RegularBottleneck(64, padding=1)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(64, 16, padding=1)
        self.regular5_1 = RegularBottleneck(16, padding=1)
        self.transposed_conv = nn.ConvTranspose2d(16,num_classes,kernel_size=2,stride=2,padding=0,bias=False)
  
    def forward(self, x):
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        
        # Stage 4 - Decoder
        d_x = self.upsample4_0(x)
        d_x = self.regular4_1(d_x)
        d_x = self.regular4_2(d_x)

        # Stage 5 - Decoder
        d_x = self.upsample5_0(d_x)
        d_x = self.regular5_1(d_x)
        y = self.transposed_conv(d_x)
        return y
if __name__ == '__main__':
    net = ENet(num_classes=19)
    data = torch.ones([1, 3, 512, 1024])
    _ = net(Variable(data))
