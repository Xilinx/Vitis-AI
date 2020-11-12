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


import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='fpn', help='model name (default: fpn)')
        parser.add_argument('--backbone', type=str, default='resnet18',choices=['resnet18', 'mobilenetv2'], \
                             help='backbone name (default: resnet18)')
        parser.add_argument('--dataset', type=str, default='citys',help='dataset name (default: cityscapes)')
        parser.add_argument('--num-classes', type=int, default=19, help='the classes numbers (default: 19 for cityscapes)')
        parser.add_argument('--data-folder', type=str, default='./data/cityscapes',help='training dataset folder (default: ./data)')
        parser.add_argument('--ignore_label', type=int, default=-1, help='the ignore label (default: 255 for cityscapes)')

        parser.add_argument('--workers', type=int, default=16, metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=1024, help='the shortest image size')
        parser.add_argument('--crop-size', type=int, default=512, help='input size for inference')
        parser.add_argument('--test-batch-size', type=int, default=10,metavar='N', help='input batch size for testing (default: 10)')
        parser.add_argument('--batch-size', type=int, default=10,metavar='N', help='input batch size for training (default: 10)')
        parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
        parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
        parser.add_argument('--train-split', type=str, default='train', help='dataset train split (default: train)')
        parser.add_argument('--ckpt_path', type=str, default='checkpoint', help='weight save directory (default: checkpoint)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',help='learning rate scheduler (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='w-decay (default: 1e-4)')
        
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
        # checking point
        parser.add_argument('--weight', type=str, default=None, help='path to final weight')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default=False, help='evaluating mIoU')
        # test option
        parser.add_argument('--test-folder', type=str, default=None, help='path to demo folder')
        parser.add_argument('--scale', type=float, default=0.5, help='downsample scale')
        # dist
        parser.add_argument('--local_rank', default=0, type=int, help='process rank on node')
        parser.add_argument('--ngpu', default=1, type=int, help='the number of gpu')
        parser.add_argument('--save-dir', type=str, default='./data/demo')
        parser.add_argument('--quant_mode', type=int, default=1)
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        return args

