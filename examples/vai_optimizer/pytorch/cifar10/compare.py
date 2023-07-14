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

from data import get_dataloader
from net import MyNet
from utils import *

from pytorch_nndct import get_pruning_runner

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
parser.add_argument(
    '--sparsity', type=float, default=0.5, help='Sparsity ratio')
parser.add_argument(
    '--save_dir',
    type=str,
    default='./',
    help='Where to save retrained model')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./dataset/cifar10',
    help='Dataset directory')
parser.add_argument(
    '--num_workers',
    type=int,
    default=48,
    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
args, _ = parser.parse_known_args()

device = 'cuda'
gpus = get_gpus(args.gpus)


if __name__ == '__main__':
  sparse_model_path = os.path.join(args.save_dir, 'mynet_sparse.pth')
  assert os.path.exists(sparse_model_path), "No sparse model!"
  slim_model_path = os.path.join(args.save_dir, 'mynet_slim.pth')
  assert os.path.exists(sparse_model_path), "No slim model!"

  if os.path.exists(args.data_dir):
    download = False
  else:
    download = True
  batch_size = args.batch_size * len(gpus)
  val_loader = get_dataloader(args.data_dir, batch_size, num_workers=args.num_workers, shuffle=False, train=False, download=download)
  sparse_model = MyNet()
  sparse_model = load_weights(sparse_model, sparse_model_path)
  input_signature = torch.randn([1, 3, 32, 32], dtype=torch.float32)
  input_signature = input_signature.to(device)
  sparse_model = sparse_model.to(device)
  pruning_runner = get_pruning_runner(sparse_model, input_signature, 'iterative')

  slim_model = pruning_runner.prune(removal_ratio=args.sparsity, mode='slim')
  slim_model = load_weights(slim_model, slim_model_path)
  slim_model = torch.nn.DataParallel(slim_model, device_ids=gpus)
  sparse_model = torch.nn.DataParallel(sparse_model, device_ids=gpus)
  criterion = torch.nn.CrossEntropyLoss().cuda()
  acc1_sparse, acc5_sparse = evaluate(val_loader, sparse_model, criterion)
  print('Accuracy of sparse model: acc1={}, acc5={}'.format(acc1_sparse, acc5_sparse))
  acc1_slim, acc5_slim = evaluate(val_loader, slim_model, criterion)
  print('Accuracy of slim model: acc1={}, acc5={}'.format(acc1_slim, acc5_slim))
  assert acc1_sparse==acc1_slim and acc5_sparse==acc5_slim
  print('Done!')
