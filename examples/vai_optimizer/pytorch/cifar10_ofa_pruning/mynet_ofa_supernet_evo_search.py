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
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_nndct import OFAPruner

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
# ofa config
parser.add_argument(
    '--channel_divisible', type=int, default=2, help='make channel divisible')
parser.add_argument(
    '--expand_ratio', type=list, default=[0.5, 0.75, 1], help='expand ratio')
parser.add_argument(
    '--excludes',
    type=list,
    default=None,
    help='excludes module')

# evo_search config
parser.add_argument(
    '--evo_search_parent_popu_size',
    type=int,
    default=16,
    help='evo search parent popu size')
parser.add_argument(
    '--evo_search_mutate_size',
    type=int,
    default=8,
    help='evo search mutate size')
parser.add_argument(
    '--evo_search_crossover_size',
    type=int,
    default=4,
    help='evo search crossover size')
parser.add_argument(
    '--evo_search_mutate_prob',
    type=float,
    default=0.2,
    help='evo search mutate prob')
parser.add_argument(
    '--evo_search_evo_iter', type=int, default=10, help='evo search evo iter')
parser.add_argument(
    '--evo_search_step', type=int, default=10, help='evo search step')
parser.add_argument(
    '--evo_search_targeted_min_macs',
    type=int,
    default=200,
    help='evo search targeted_min_macs')
parser.add_argument(
    '--evo_search_targeted_max_macs',
    type=int,
    default=300,
    help='evo search targeted_max_macs')

parser.add_argument(
    '--pretrained_ofa_model',
    type=str,
    default='mynet_pruned.pth',
    help='Pretrained ofa model filepath')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./dataset/cifar10',
    help='Dataset directory')
parser.add_argument(
    '--num_workers',
    type=int,
    default=16,
    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')

args, _ = parser.parse_known_args()

def get_gpus(device):
  return [int(i) for i in device.split(',')]

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

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

def eval_fn(model, dataloader_test):
  top1 = AverageMeter('Acc@1', ':6.2f')
  model.eval()
  with torch.no_grad():
    for i, (images, targets) in enumerate(dataloader_test):
      images = images.cuda()
      targets = targets.cuda()
      outputs = model(images)
      acc1, _ = accuracy(outputs, targets, topk=(1, 5))
      top1.update(acc1[0], images.size(0))
  return float(top1.avg)

def calibration_fn(model, train_loader, number_forward=16):
  model.eval()
  for n, m in model.named_modules():
    if isinstance(m, torch.nn.BatchNorm2d):
      m.training = True
      m.momentum = None
      m.reset_running_stats()
  print("Calibration BN start...")
  with torch.no_grad():
    for index, (images, _) in enumerate(train_loader):
      images = images.cuda()
      model(images)
      if index > number_forward:
        break
  print("Calibration BN end...")

class MyNet(nn.Module):
  def __init__(self):
    super(MyNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(32, 128, 3, stride=2)
    self.bn2 = nn.BatchNorm2d(128)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv3 = nn.Conv2d(128, 256, 3)
    self.bn3 = nn.BatchNorm2d(256)
    self.relu3 = nn.ReLU(inplace=True)
    self.conv4 = nn.Conv2d(256, 512, 3)
    self.bn4 = nn.BatchNorm2d(512)
    self.relu4 = nn.ReLU(inplace=True)
    self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
    self.fc1 = nn.Linear(512, 32)
    self.fc = nn.Linear(32, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu3(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = self.relu4(x)
    x = self.avgpool1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.fc(x)
    return x

if __name__ == '__main__':

  gpus = get_gpus(args.gpus)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = MyNet()

  transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

  val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

  model.to(device)

  criterion = torch.nn.CrossEntropyLoss().cuda()

  input_signature = torch.randn([1, 3, 32, 32], dtype=torch.float32).cuda()

  ofa_pruner = OFAPruner(model, input_signature)
  ofa_model = ofa_pruner.ofa_model(args.expand_ratio, args.channel_divisible,
                                   args.excludes)

  model = ofa_model.cuda()

  #reloading model
  with open(args.pretrained_ofa_model, 'rb') as f:
    checkpoint = torch.load(f, map_location='cpu')
  assert isinstance(checkpoint, dict)
  for k, v in model.state_dict().items():
    v.copy_(checkpoint[k])

  dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
      model, 'max')
  static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
      dynamic_subnet, dynamic_subnet_setting)

  max_subnet_macs = macs

  dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
      model, 'min')
  static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
      dynamic_subnet, dynamic_subnet_setting)

  min_subnet_macs = macs

  print('max subnet macs(M):', max_subnet_macs)
  print('min subnet macs(M):', min_subnet_macs)

  targeted_min_macs = args.evo_search_targeted_min_macs
  targeted_max_macs = args.evo_search_targeted_max_macs

  assert targeted_min_macs <= targeted_max_macs and min_subnet_macs <= targeted_min_macs and targeted_max_macs <= max_subnet_macs

  pareto_global = ofa_pruner.run_evolutionary_search(
      model, calibration_fn, (train_loader,), eval_fn, (val_loader,), 'acc@top1', 'max',
      targeted_min_macs, targeted_max_macs, args.evo_search_step,
      args.evo_search_parent_popu_size, args.evo_search_evo_iter,
      args.evo_search_mutate_size, args.evo_search_mutate_prob,
      args.evo_search_crossover_size)

  ofa_pruner.save_subnet_config(pareto_global, 'cifar10_pareto_global.txt')
