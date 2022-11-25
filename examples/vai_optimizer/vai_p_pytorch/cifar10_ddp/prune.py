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
