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

import os
import argparse
import torch

class Configs():
    def __init__(self):

        parser = argparse.ArgumentParser("face_quality")
    
        #dataset options
        parser.add_argument('--dataset_root', type=str, default='../../data/face_quality', help='path to dataset')
        parser.add_argument('--anno_train_list', type=str, default='../../data/face_quality/train_list.txt', help='path to groundtruth file')
        parser.add_argument('--anno_test_list', type=str, default='../../data/face_quality/test_list.txt', help='path to groundtruth file')
        parser.add_argument('--size', nargs='+', type=int, default=[80, 60], help='input size')
        parser.add_argument('--pointsNum', type=int, default=5, help='The number of points')
        #training options
        parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
        parser.add_argument('--test_batch_size', type=int, default=100, help='test batch size')
        parser.add_argument('--pretrained', default='../../float/points_quality_80x60_addqneg_nodrop_gray3_1.94_12.2.pth', type=str, metavar='PATH',help='path to pretrained (default: none)')
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        parser.add_argument('--epochs', type=int, default=30, help='total epochs')
        parser.add_argument('--base_lr', type=int, default=5e-4, help='base learning rate')
        parser.add_argument('--train_worker', type=int, default=1, help='number of train data loading workers')
        parser.add_argument('--test_worker', type=int, default=1, help='number of test data loading workers')
        parser.add_argument('--mean', type=float, default=[0.5, 0.5, 0.5])
        parser.add_argument('--std', type=float, default=[0.5, 0.5, 0.5])
        parser.add_argument('--milestones', type=int, default=[10, 15, 20, 25], help='Epoches at which learning rate decays.')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
        parser.add_argument('--gamma', type=float, default=0.1, help='gamma value')
        parser.add_argument('--print_freq', type=int, default=100, help='print frequency')

        #testing options
        parser.add_argument('--visual_test_list', type=str, default='../test/visual_test_list.txt', help='path to visualize test file')
        #gpu options
        parser.add_argument('--gpu', type=int, default=0, help='gpu id')

        #quantization options
        parser.add_argument('--quant_mode', default='calib', choices=['float', 'calib', 'test'], \
                                            help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
        parser.add_argument('--dump_xmodel', dest='dump_xmodel', action='store_true', help='dump xmodel after test')
        parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'], help='assign runtime device')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

