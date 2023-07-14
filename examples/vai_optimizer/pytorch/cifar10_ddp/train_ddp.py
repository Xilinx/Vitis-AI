import argparse
import os

import torch

from data import get_dataloader_ddp, get_dataloader
from net import MyNet
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--local_rank', type=int, default=0, help='node rank for distributed training')
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
parser.add_argument(
    '--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--epochs', type=int, default=1, help='Train epoch')
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
    default=64,
    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument(
    '--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')

args, _ = parser.parse_known_args()


device = 'cuda'
gpus = get_gpus(args.gpus)

if __name__ == '__main__':
  torch.distributed.init_process_group(backend="nccl")
  torch.cuda.set_device(args.local_rank)
  batch_size = args.batch_size * len(gpus)
  model = MyNet()
  model.cuda()
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

  if os.path.exists(args.data_dir):
    download = False
  else:
    download = True

  train_loader = get_dataloader_ddp(args.data_dir, batch_size, num_workers=args.num_workers, shuffle=True, train=True, download=download)
  val_loader = get_dataloader_ddp(args.data_dir, batch_size, num_workers=args.num_workers, shuffle=False, train=False, download=download)

  criterion = torch.nn.CrossEntropyLoss().cuda()
  if not os.path.exists(args.pretrained):
    optimizer = torch.optim.Adam(
      model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_acc1 = 0
    for epoch in range(args.epochs):
      train(train_loader, model, criterion, optimizer, epoch)
      lr_scheduler.step()
      acc1, acc5 = evaluate(val_loader, model, criterion)
      if acc1 > best_acc1:
        best_acc1 = acc1
        torch.save(model.state_dict(), args.pretrained)
