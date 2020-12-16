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

import argparse
import torch
import time
import logging
import os, sys
sys.path.insert(0,'..')
from utils.utils import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description='pspnet with multi-datasets')
        parser.add_argument('--model_path', default=None,type=str, help='model_path')
        parser.add_argument('--normlayer', default='BN',type=str, help='normlayer type')
        parser.add_argument('--index', default=1, type=int, help='semi id numbers')
        parser.add_argument('--data-set', default='cityscape',type=str, metavar='', help='')
        parser.add_argument('--data-dir', type=str, default='/scratch/workspace/wangli/Dataset/', help="Path to the dataset.")
        parser.add_argument('--ignore-index', default=255, type=int, help='ignore_index of the dataset')        

        parser.add_argument('--classes-num', default=19, type=int,metavar='N', help='class num of the dataset')
        #extra dataset
        parser.add_argument('--loss-w', default=1.0, type=float,metavar='N', help='loss weight')

        parser.add_argument('--resume', default='True', type=str2bool, metavar='is or not use student', help='is or not use student ckpt')
        parser.add_argument('--ckpt-path', default='',type=str, metavar='student ckpt path', help='student ckpt path')
        parser.add_argument("--batch-size", type=int, default=8, help="Number of images sent to the network in one step.")
        parser.add_argument('--start_epoch', default=0, type=int,metavar='start_epoch', help='start_epoch')
        parser.add_argument('--parallel', default='True', type=str, metavar='parallel', help='attribute of saved name')
        parser.add_argument("--input-size", type=str, default='512,512', help="Comma-separated string with height and width of images.")

        parser.add_argument("--is-training", action="store_true", help="Whether to updates the running means and variances during the training.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--num-steps", type=int, default=40000, help="Number of training steps.")
        parser.add_argument("--save-steps", type=int, default=10000, help="Number of save steps.")
        parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
        parser.add_argument("--random-mirror", action="store_true", help="Whether to randomly mirror the inputs during the training.")
        parser.add_argument("--warmup", action="store_true", help="warmup")
        parser.add_argument("--random-scale", action="store_true", help="Whether to randomly scale the inputs during the training.")
        parser.add_argument("--weight-decay", type=float, default=1.0e-4, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--gpu", type=str, default='None', help="choose gpu device.")
        parser.add_argument("--recurrence", type=int, default=1, help="choose the number of recurrence.")

        parser.add_argument("--last-step", type=int, default=0, help="last train step.")
        parser.add_argument("--is-load-imgnet", type=str2bool, default='True', help="is student load imgnet")
        parser.add_argument("--pretrain-model-imgnet", type=str, default='None', help="student pretrain model on imgnet")
        parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for G")
        args = parser.parse_args()

        args.log_path = os.path.join(args.ckpt_path, 'logs')

        log_init(args.log_path, args.data_set)
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.gpu_num = len(args.gpu.split(','))
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        logger_path = args.log_path + '/tensorboard/'
        return args


class TrainOptionsForTest():
    def initialize(self):
        parser = argparse.ArgumentParser(description='semantic segmentation')
        parser.add_argument("--data-dir", type=str, default='', help="")
        parser.add_argument("--resume-from", type=str, default='', help="")
        args = parser.parse_args()
        return args
