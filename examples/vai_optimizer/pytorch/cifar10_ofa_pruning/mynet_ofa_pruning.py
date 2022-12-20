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
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pytorch_nndct import OFAPruner

parser = argparse.ArgumentParser()
parser.add_argument(
    '--lr', type=float, default=1e-2, help='Initial learning rate')

# ofa config
parser.add_argument(
    '--channel_divisible', type=int, default=2, help='make channel divisible')
parser.add_argument(
    '--expand_ratio', type=list, default=[0.5, 0.75, 1], help='expand ratio')
parser.add_argument(
    '--excludes', type=list, default=None, help='excludes module')

parser.add_argument('--epoches', type=int, default=20, help='Train epoch')
parser.add_argument(
    '--pretrained',
    type=str,
    default='mynet_trained.pth',
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
    self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
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

class ProgressMeter(object):

  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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

def adjust_learning_rate(optimizer, epoch, lr):
  """Sets the learning rate to the initial LR decayed by every 2 epochs"""
  lr = lr * (0.1**(epoch // 2))

  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def train(train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          teacher_model=None,
          ofa_pruner=None,
          soft_criterion=None):
  batch_time = AverageMeter('Time', ':6.3f')
  data_time = AverageMeter('Data', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(train_loader), [batch_time, data_time, losses, top1, top5],
      prefix="Epoch: [{}]".format(epoch))

  # switch to train mode
  model.train()

  end = time.time()
  for i, (images, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    model = model.cuda()
    images = images.cuda()
    target = target.cuda()

    # total subnets to be sampled
    optimizer.zero_grad()

    if teacher_model:
      teacher_model.train()
      with torch.no_grad():
        soft_logits = teacher_model(images).detach()

    if ofa_pruner:
      for arch_id in range(4):
        if arch_id == 0:
          model, _ = ofa_pruner.sample_subnet(model, 'max')
        elif arch_id == 1:
          model, _ = ofa_pruner.sample_subnet(model, 'min')
        else:
          model, _ = ofa_pruner.sample_subnet(model, 'random')

        # calcualting loss
        output = model(images)

        if soft_criterion:
          loss = soft_criterion(output, soft_logits) + criterion(output, target)
        else:
          loss = criterion(output, target)

        loss.backward()
    else:
      # compute output
      output = model(images)
      loss = criterion(output, target)
      loss.backward()

    optimizer.step()

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), images.size(0))
    top1.update(acc1[0], images.size(0))
    top5.update(acc5[0], images.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % 100 == 0:
      progress.display(i)

def evaluate(dataloader, model, criterion, ofa_pruner=None, train_loader=None):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

  if ofa_pruner:
    dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
        model, 'max')
    static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
        dynamic_subnet, dynamic_subnet_setting)

    model = static_subnet.cuda()

    with torch.no_grad():
      model.eval()
      ofa_pruner.reset_bn_running_stats_for_calibration(model)

      print('runing bn calibration...')

      for batch_idx, (images, _) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        model(images)
        if batch_idx >= 16:
          break

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(dataloader):
      model = model.cuda()
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)

      # compute output
      output = model(images)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))

  return top1.avg, top5.avg

def load_weights(model, model_path):
  checkpoint = torch.load(model_path)
  model.load_state_dict(checkpoint)
  return model

def get_dataloader():
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  train_dataset = datasets.CIFAR10(
      root=args.data_dir, train=True, download=True, transform=transform)
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers)

  val_dataset = datasets.CIFAR10(
      root=args.data_dir, train=False, download=True, transform=transform)
  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers)

  return train_loader, val_loader

if __name__ == '__main__':
  model = MyNet()
  model.cuda()

  print(model)

  train_loader, val_loader = get_dataloader()
  criterion = torch.nn.CrossEntropyLoss().cuda()
  if os.path.exists(args.pretrained):
    ckpt = torch.load(args.pretrained)
    model.load_state_dict(ckpt)
    acc1, acc5 = evaluate(val_loader, model, criterion)
  else:
    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay)
    best_acc1 = 0
    for epoch in range(args.epoches):
      #adjust_learning_rate(optimizer, epoch, args.lr)
      train(train_loader, model, criterion, optimizer, epoch)
      acc1, acc5 = evaluate(val_loader, model, criterion)
      if acc1 > best_acc1:
        best_acc1 = acc1
        torch.save(model.state_dict(), args.pretrained)

  teacher_model = model
  inputs = torch.randn([1, 3, 32, 32], dtype=torch.float32).cuda()

  ofa_pruner = OFAPruner(teacher_model, inputs)
  ofa_model = ofa_pruner.ofa_model(args.expand_ratio, args.channel_divisible,
                                   args.excludes)

  model = ofa_model.cuda()

  optimizer = torch.optim.Adam(
      model.parameters(), args.lr, weight_decay=args.weight_decay)

  soft_criterion = None

  best_acc1 = 0
  for epoch in range(args.epoches):
    train(train_loader, model, criterion, optimizer, epoch, teacher_model,
          ofa_pruner, soft_criterion)
    acc1, acc5 = evaluate(val_loader, model, criterion, ofa_pruner,
                          train_loader)
    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if is_best:
      torch.save(model.state_dict(), 'mynet_pruned.pth')
