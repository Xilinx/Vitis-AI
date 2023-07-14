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

import argparse
import os
import time

import torch

from common import AverageMeter, ProgressMeter
from data import get_dataloader, get_subnet_dataloader
from net import MyNet
from utils import *

from pytorch_nndct import get_pruning_runner

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
parser.add_argument(
    '--subset_len',
    default=None,
    help='Subset length for evaluating model in analysis, using the whole validation dataset if it is not set'
)
parser.add_argument(
    '--num_subnet', type=int, default=20, help='The number of subnets searched')
parser.add_argument(
    '--sparsity', type=float, default=0.5, help='Sparsity ratio')
parser.add_argument(
    '--pretrained',
    type=str,
    default='mynet.pth',
    help='Pretrained model filepath')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./dataset/cifar10',
    help='Dataset directory')
parser.add_argument(
    '--num_workers',
    type=int,
    default=1,
    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
args, _ = parser.parse_known_args()

device = 'cuda'
gpus = get_gpus(args.gpus)


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions
    for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def eval_fn(model, dataloader):
  top1 = AverageMeter('Acc@1', ':6.2f')
  model.eval()
  with torch.no_grad():
    for i, (images, targets) in enumerate(dataloader):
      images = images.cuda()
      targets = targets.cuda()
      outputs = model(images)
      acc1, _ = accuracy(outputs, targets, topk=(1, 5))
      top1.update(acc1[0], images.size(0))
  return top1.avg

def calibration_fn(model, dataloader, number_forward=100):
  model.train()
  print("Adaptive BN atart...")
  with torch.no_grad():
    for index, (images, target) in enumerate(dataloader):
      images = images.cuda()
      model(images)
      if index > number_forward:
        break
  print("Adaptive BN end...")

if __name__ == '__main__':
  assert os.path.exists(args.pretrained), "No pretrained model!"
  if os.path.exists(args.data_dir):
    download = False
  else:
    download = True
  
  if args.subset_len:
    data_loader = get_subnet_dataloader(args.data_dir, batch_size, args.subset_len, num_workers=args.num_workers, shuffle=False, train=False, download=download)
  else:
    data_loader = get_dataloader(args.data_dir, args.batch_size, num_workers=args.num_workers, shuffle=False, train=False, download=download)

  model = MyNet()
  model = load_weights(model, args.pretrained)
  input_signature = torch.randn([1, 3, 32, 32], dtype=torch.float32)
  input_signature = input_signature.to(device)
  model = model.to(device)
  pruning_runner = get_pruning_runner(model, input_signature, 'one_step')

  pruning_runner.search(
      gpus=gpus,
      calibration_fn=calibration_fn,
      calib_args=(data_loader,),
      num_subnet=args.num_subnet,
      removal_ratio=args.sparsity,
      eval_fn=eval_fn,
      eval_args=(data_loader,))
