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

class Configs():
    def __init__(self):

        parser = argparse.ArgumentParser("ENet(modified) on Cityscapes")
    
        #dataset options
        parser.add_argument('--dataset', type=str, default='cityscapes', help='dataset name')
        parser.add_argument('--data_root', type=str, default='./data/cityscapes', help='path to dataset')
        parser.add_argument('--num_classes', type=int, default=19, help='classes numbers')
        parser.add_argument('--ignore_label', type=int, default=255, help='ignore index')

        parser.add_argument('--checkpoint_dir', type=str, default='ckpt-cityscapes', help='path to checkpoint')
        parser.add_argument('--input_size', nargs='+', type=int, default=[1024, 512], help='input size')
        parser.add_argument('--resume', action='store_true', help='wether training with resume')
        parser.add_argument('--weight', type=str, default=None, help='resume from weight')
        #training options
        parser.add_argument('--train_batch_size', type=int, default=20, help='batch size')
        parser.add_argument('--total_epoch', type=int, default=300, help='total epochs')
        parser.add_argument('--lr_patience', type=int, default=100, help='lr patience')
        parser.add_argument('--lr', type=float, default=0.01, help='init learning rate')
        parser.add_argument('--momentum', type=float, default=0.99, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
        parser.add_argument('--lr_decay', type=float, default=0.9, help='lr decay')
        parser.add_argument('--test_only', action='store_true', help='if only test the trained model')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--gpu_num', type=int, default=1, help='number of gpus')
        parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
        parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
        # cuda, seed and logging
        parser.add_argument('--cuda', action='store_true', default=True, help='with CUDA training')
        #validation options
        parser.add_argument('--val_batch_size', type=int, default=1, help='batch size')
        # evaluation miou options
        parser.add_argument('--eval', action='store_true', help='evaluation miou mode')
        # demo options
        parser.add_argument('--demo_dir', type=str, default='./data/demo', help='path to demo dataset')
        parser.add_argument('--save_dir', type=str, default='./data/demo/results', help='path to save demo prediction')
        parser.add_argument('--quant_mode', type=int, default=1)
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = args.cuda and torch.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.total_epoch is None:
            total_epoch = {
                'cityscapes': 300,
            }
            args.total_epoch = total_epoch[args.dataset.lower()]

        if args.lr is None:
            lrs = {
                'cityscapes': 0.01,
            }
            args.lr = lrs[args.dataset.lower()]
        print(args)
        return args

