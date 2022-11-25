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

import torch

from net import MyNet
from utils import *

from pytorch_nndct import get_pruning_runner

parser = argparse.ArgumentParser()
parser.add_argument(
    '--sparsity', type=float, default=0.5, help='Sparsity ratio')
parser.add_argument(
    '--save_dir',
    type=str,
    default='./',
    help='Where to save retrained model')
args, _ = parser.parse_known_args()

device = 'cuda'

if __name__ == '__main__':
  sparse_model_path = os.path.join(args.save_dir, 'mynet_sparse.pth')
  assert os.path.exists(sparse_model_path), "No sparse model!"

  slim_model_path = os.path.join(args.save_dir, 'mynet_slim.pth')
  sparse_model = MyNet()
  sparse_model = load_weights(sparse_model, sparse_model_path)
  sparse_model.to(device)
  input_signature = torch.randn([1, 3, 32, 32], dtype=torch.float32)
  input_signature = input_signature.to(device)

  pruning_runner = get_pruning_runner(sparse_model, input_signature, 'iterative')
  slim_model = pruning_runner.prune(removal_ratio=args.sparsity, mode='slim')
  torch.save(slim_model.state_dict(), slim_model_path)
  print('Convert sparse model to slim model done!')
