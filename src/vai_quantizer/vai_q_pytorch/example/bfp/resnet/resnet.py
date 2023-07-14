# Copyright 2023 Xilinx Inc.
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
#

# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import shutil
import time

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn

from pytorch_nndct.nn.modules import functional
from pytorch_nndct.quantization import bfp

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', default='resnet50', type=str)
parser.add_argument(
    '--data_dir',
    default='/group/dataset/imagenet/pytorch',
    help='Data set directory.')
parser.add_argument(
    '--config_file',
    default='config.yaml',
    type=str,
    help='Configuration file path.')
parser.add_argument('--epochs', default=0, type=int, help='between 1-10')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=1e-6,
    type=float,
    metavar='LR',
    help='initial learning rate',
    dest='lr')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--wd',
    '--weight-decay',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)',
    dest='weight_decay')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument(
    '--val_freq_iters',
    default=80000,
    type=int,
    help='Run validation every n iters during training.')
parser.add_argument('--num', default=10, type=int, help='count')
parser.add_argument('--optim', default='AdamW', type=str, help='SGD or AdamW')
parser.add_argument(
    '--pretrained',
    default="/group/modelzoo/compression/pretrained/resnet50.pth",
    type=str,
    help='Directory of pretrained weights')
parser.add_argument('--acc', default=76.012, type=int, help='fp32')
parser.add_argument(
    '--val_display_freq',
    default=10,
    type=int,
    help='Display frequency of validation.')
parser.add_argument(
    '--train_display_freq',
    default=50,
    type=int,
    help='Display frequency of training.')
parser.add_argument(
    '--model_dir',
    default='checkpoints',
    type=str,
    help='Directory to save trained weights.')
args = parser.parse_args()

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(
      in_planes,
      out_planes,
      kernel_size=3,
      stride=stride,
      padding=dilation,
      groups=groups,
      bias=False,
      dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
  """1x1 convolution"""
  return nn.Conv2d(
      in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               downsample=None,
               groups=1,
               base_width=64,
               dilation=1,
               norm_layer=None):
    super(Bottleneck, self).__init__()

    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    width = int(planes * (base_width / 64.)) * groups

    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv1x1(inplanes, width)
    self.bn1 = norm_layer(width)
    self.conv2 = conv3x3(width, width, stride, groups, dilation)
    self.bn2 = norm_layer(width)
    self.conv3 = conv1x1(width, planes * self.expansion)
    self.bn3 = norm_layer(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

    self.add = functional.Add()

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    #out = out.bfloat16().float() + identity.bfloat16().float()
    out = self.add(out, identity)
    out = self.relu(out)

    return out

class ResNet(nn.Module):

  def __init__(self,
               block,
               layers,
               num_classes=1000,
               zero_init_residual=False,
               groups=1,
               width_per_group=64,
               replace_stride_with_dilation=None,
               norm_layer=None):
    super().__init__()

    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer
    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError(
          "replace_stride_with_dilation should be None "
          "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    self.groups = groups
    self.base_width = width_per_group

    # conv1
    self.conv1 = nn.Conv2d(
        3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    # conv2_x
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    # conv3_x
    self.layer2 = self._make_layer(
        block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    # conv4_x
    self.layer3 = self._make_layer(
        block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    # conv5_x
    self.layer4 = self._make_layer(
        block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
    # avgpool, fc
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes * block.expansion, stride),
          norm_layer(planes * block.expansion))

    layers = []
    layers.append(
        block(self.inplanes, planes, stride, downsample, self.groups,
              self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              groups=self.groups,
              base_width=self.base_width,
              dilation=self.dilation,
              norm_layer=norm_layer))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)

    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    #x = self.avgpool(x.bfloat16().float())
    x = self.avgpool(x)

    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x

def resnet50(**kwargs):

  return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):

  return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):

  return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def load_imagenet(batch_size):
  # Data loading code
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  train_dir = os.path.join(args.data_dir, 'train')
  val_dir = os.path.join(args.data_dir, 'val')

  train_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(
          train_dir,
          transforms.Compose([
              transforms.RandomResizedCrop(224),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize,
          ])),
      batch_size=batch_size,
      shuffle=True,
      num_workers=4,
      pin_memory=True)

  val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(
          val_dir,
          transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              normalize,
          ])),
      batch_size=batch_size,
      shuffle=False,
      num_workers=4,
      pin_memory=True)

  return train_loader, val_loader

def train(train_loader, val_loader, model, criterion, optimizer, epoch, device,
          best_acc1, args):
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

    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # compute output
    output = model(images)
    loss = criterion(output, target)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), images.size(0))
    top1.update(acc1.item(), images.size(0))
    top5.update(acc5.item(), images.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.train_display_freq == 0:
      progress.display(i)

    if i != 0 and i % args.val_freq_iters == 0:
      # evaluate on validation set
      acc1 = validate(val_loader, model, criterion, device, args)

      # remember best acc@1 and save checkpoint
      is_best = acc1 > best_acc1
      best_acc1 = max(acc1, best_acc1)

      print('best_acc1: {a:.3f} \n'.format(a=best_acc1))
      print('best_acc1 normalized: {b:.3f} \n'.format(b=best_acc1 / args.acc))

      save_checkpoint(
          {
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'best_acc1': best_acc1,
              'optimizer': optimizer.state_dict(),
          }, is_best, args)

  return best_acc1

def adjust_learning_rate(optimizer, epoch, args):
  """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
  lr = args.lr * (0.1**(epoch // 10))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def validate(val_loader, model, criterion, device, args):

  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
      images = images.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)

      # compute output
      output = model(images)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))

      losses.update(loss.item(), images.size(0))
      top1.update(acc1.item(), images.size(0))
      top5.update(acc5.item(), images.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % args.val_display_freq == 0:
        progress.display(i)

    print(
        ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} '.format(
            top1=top1, top5=top5),
        flush=True)
    print(
        ' * Normalized_Acc@1 {res:.3f} \n'.format(res=top1.avg / args.acc),
        flush=True)

  return top1.avg

def save_checkpoint(state, is_best, args):
  ckpt_path = os.path.join(args.model_dir, 'ckpt.pth')
  torch.save(state, ckpt_path)
  if is_best:
    shutil.copyfile(ckpt_path, os.path.join(args.model_dir, 'best_ckpt.pth'))

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
    print('\t'.join(entries), flush=True)

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #device = torch.device('cpu')

  # create model
  if args.model == 'resnet50':
    model = resnet50()
    args.acc = 76.012
  elif args.model == 'resnet101':
    model = resnet101()
    args.acc = 77.314
  elif args.model == 'resnet152':
    model = resnet152()
    args.acc = 78.250
  else:
    raise NotImplementedError('Model not implemented')

  model = model.to(device)
  model.load_state_dict(torch.load(args.pretrained))

  # data loading
  train_loader, val_loader = load_imagenet(args.batch_size)
  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss().to(device)

  inputs = torch.randn([args.batch_size, 3, 224, 224],
                       dtype=torch.float32,
                       device=device)
  # Two built-in data types: mx6 and mx9
  model = bfp.quantize_model(model, inputs, dtype='mx9')
  # Or you can use config file
  # model = bfp.quantize_model(model, inputs, config_file='bfp.yaml')

  best_acc1 = validate(val_loader, model, criterion, device, args)

  if args.optim == 'SGD':
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
  elif args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False)
  else:
    raise NotImplementedError('Optim not implemented')

  # training
  for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    with torch.autograd.set_detect_anomaly(True):
      best_acc1 = train(train_loader, val_loader, model, criterion, optimizer,
                        epoch, device, best_acc1, args)

  if args.epochs > 0:
    validate(val_loader, model, criterion, device, args)
