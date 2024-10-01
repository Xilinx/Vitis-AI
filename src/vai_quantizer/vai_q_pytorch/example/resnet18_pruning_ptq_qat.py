#
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
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch_nndct import get_pruning_runner
from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional

# pruning config
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
parser.add_argument(
    '--method', type=str, default='iterative', help='iterative/one_step')
parser.add_argument(
    '--load_slim_model', action='store_true', help='Load slim model')
parser.add_argument(
    '--slim_model',
    type=str,
    default='resnet18_pruned_best.pth',
    help='Slim/Pruned model filepath')

parser.add_argument(
    '--pruning_lr', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument(
    '--sparsity', type=float, default=0.1, help='Sparsity ratio')
parser.add_argument(
    '--ana_subset_len',
    type=int,
    default=2000,
    help='Subset length for evaluating model in analysis, using the whole validation dataset if it is not set'
)
parser.add_argument(
    '--num_subnet', type=int, default=20, help='Total number of subnet')
parser.add_argument('--epochs', type=int, default=1, help='Train epoch')
parser.add_argument(
    '--pretrained',
    type=str,
    default='/torchvision_model/resnet18-f37072fd.pth',
    help='Pretrained model filepath')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./dataset/imagenet/raw-data',
    help='Dataset directory')
parser.add_argument(
    '--num_workers',
    type=int,
    default=16,
    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
parser.add_argument(
    '--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument(
    '--channel_divisible', type=int, default=2, help='channel_divisible')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')

parser.add_argument(
    '--save_dir',
    default='./pruning_qat_models',
    help='Directory to save trained models.')

# ptq config
parser.add_argument(
    '--quant_mode',
    default='calib',
    choices=['float', 'calib', 'test'],
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model'
)

# qat config
parser.add_argument(
    '--qat_quantizer_lr',
    default=1e-2,
    type=float,
    help='Initial learning rate of quantizer.')
parser.add_argument(
    '--qat_quantizer_lr_decay',
    default=0.5,
    type=int,
    help='Learning rate decay ratio of quantizer.')
parser.add_argument(
    '--qat_weight_lr',
    default=1e-5,
    type=float,
    help='Initial learning rate of network weights.')
parser.add_argument(
    '--qat_weight_lr_decay',
    default=0.94,
    type=int,
    help='Learning rate decay ratio of network weights.')
parser.add_argument(
    '--display_freq',
    default=100,
    type=int,
    help='Display training metrics every n steps.')
parser.add_argument(
    '--val_freq', default=10000, type=int, help='Validate model every n steps.')
parser.add_argument(
    '--quantizer_norm',
    default=True,
    type=bool,
    help='Use normlization for quantizer.')

args, _ = parser.parse_known_args()

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

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               downsample=None,
               groups=1,
               base_width=64,
               dilation=1,
               norm_layer=None):
    super(BasicBlock, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64:
      raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu1 = nn.ReLU(inplace=True)
    #self.relu1 = nn.functional.relu
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

    self.skip_add = functional.Add()
    self.relu2 = nn.ReLU(inplace=True)

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out = self.skip_add(out, identity)
    out = self.relu2(out)

    return out

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
    self.relu1 = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

    self.skip_add = functional.Add()
    self.relu2 = nn.ReLU(inplace=True)
    self.relu3 = nn.ReLU(inplace=True)

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    # The original code was:
    # out += identity
    # Replace '+=' with Add module cause we want to quantize add op.
    out = self.skip_add(out, identity)
    out = self.relu3(out)

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
    super(ResNet, self).__init__()
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
    self.conv1 = nn.Conv2d(
        3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(
        block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(
        block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(
        block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    self.quant_stub = nndct_nn.QuantStub()
    self.dequant_stub = nndct_nn.DeQuantStub()

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

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
          norm_layer(planes * block.expansion),
      )

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
    x = self.quant_stub(x)

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    x = self.dequant_stub(x)
    return x

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
  model = ResNet(block, layers, **kwargs)
  if pretrained:
    model.load_state_dict(torch.load(args.pretrained))
  return model

def resnet18(pretrained=False, progress=True, **kwargs):
  return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                 **kwargs)

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

def adjust_learning_rate(optimizer, epoch, step):
  """Sets the learning rate to the initial LR decayed by decay ratios"""

  weight_lr_decay_steps = 3000 * (24 / args.batch_size)
  quantizer_lr_decay_steps = 1000 * (24 / args.batch_size)

  for param_group in optimizer.param_groups:
    group_name = param_group['name']
    if group_name == 'weight' and step % weight_lr_decay_steps == 0:
      lr = args.qat_weight_lr * (
          args.qat_weight_lr_decay**(step / weight_lr_decay_steps))
      param_group['lr'] = lr
      print('Adjust lr at epoch {}, step {}: group_name={}, lr={}'.format(
          epoch, step, group_name, lr))
    if group_name == 'quantizer' and step % quantizer_lr_decay_steps == 0:
      lr = args.qat_quantizer_lr * (
          args.qat_quantizer_lr_decay**(step / quantizer_lr_decay_steps))
      param_group['lr'] = lr
      print('Adjust lr at epoch {}, step {}: group_name={}, lr={}'.format(
          epoch, step, group_name, lr))

def train_one_step(model, inputs, criterion, optimizer, step, device):
  # switch to train mode
  model.train()

  images, target = inputs
  if not isinstance(model, nn.DataParallel):
    model = model.to(device)
  images = images.to(device, non_blocking=True)
  target = target.to(device, non_blocking=True)

  # compute output
  output = model(images)
  loss = criterion(output, target)

  if hasattr(
      model if not isinstance(model, nn.DataParallel) else model,
      'quantizer_parameters'):
    l2_decay = 1e-4
    l2_norm = 0.0
    q_params = model.quantizer_parameters() if not isinstance(model, nn.DataParallel) \
      else model.module.quantizer_parameters()
    for param in q_params:
      l2_norm += torch.pow(param, 2.0)[0]
    if args.quantizer_norm:
      loss += l2_decay * torch.sqrt(l2_norm)

  # measure accuracy and record loss
  acc1, acc5 = accuracy(output, target, topk=(1, 5))

  # compute gradient and do SGD step
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss, acc1, acc5

def train(model, train_loader, val_loader, criterion, device_ids):
  best_acc1 = 0
  best_filepath = None

  if device_ids is not None and len(device_ids) > 0:
    device = f"cuda:{device_ids[0]}"
    model = model.to(device)
    if len(device_ids) > 1:
      model = nn.DataParallel(model, device_ids=device_ids)

  batch_time = AverageMeter('Time', ':6.3f')
  data_time = AverageMeter('Data', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')

  if hasattr(
      model if not isinstance(model, nn.DataParallel) else model,
      'quantizer_parameters'):
    param_groups = [{
        'params': model.quantizer_parameters() if not isinstance(model, nn.DataParallel) \
          else model.module.quantizer_parameters(),
        'lr': args.qat_quantizer_lr,
        'name': 'quantizer'
    }, {
        'params': model.non_quantizer_parameters() if not isinstance(model, nn.DataParallel) \
          else model.module.non_quantizer_parameters(),
        'lr': args.qat_weight_lr,
        'name': 'weight'
    }]

    optimizer = torch.optim.Adam(
        param_groups, args.qat_weight_lr, weight_decay=args.weight_decay)
  else:
    param_groups = [{
        'params': model.parameters() if not isinstance(model, nn.DataParallel) \
          else model.module.parameters(),
        'lr': args.pruning_lr,
        'name': 'weight'
    }]

    optimizer = torch.optim.Adam(
        param_groups, args.pruning_lr, weight_decay=args.weight_decay)

  for epoch in range(args.epochs):
    progress = ProgressMeter(
        len(train_loader) * args.epochs,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch[{}], Step: ".format(epoch))

    for i, (images, target) in enumerate(train_loader):
      end = time.time()
      # measure data loading time
      data_time.update(time.time() - end)

      step = len(train_loader) * epoch + i

      adjust_learning_rate(optimizer, epoch, step)
      loss, acc1, acc5 = train_one_step(model, (images, target), criterion,
                                        optimizer, step, device)

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      if step % args.display_freq == 0:
        progress.display(step)

      if step % args.val_freq == 0:
        # evaluate on validation set
        acc1 = evaluate(val_loader, model, criterion)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        filepath = save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) \
                  else model.module.state_dict(),
                'best_acc1': best_acc1,
            }, is_best, args.save_dir)

        if is_best:
          best_filepath = filepath
          if hasattr(model, 'slim_state_dict'):
            torch.save(model.slim_state_dict(), 'resnet18_pruned_best.pth')
          if hasattr(model, 'sparse_state_dict'):
            torch.save(model.sparse_state_dict(), 'resnet18_sparse_best.pth')

  return best_filepath

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
  return top1.avg

def calibration_fn(model, train_loader, number_forward=100):
  model.train()
  print("Adaptive BN atart...")
  with torch.no_grad():
    for index, (images, target) in enumerate(train_loader):
      images = images.cuda()
      model(images)
      if index > number_forward:
        break
  print("Adaptive BN end...")

def mkdir_if_not_exist(x):
  if not x or os.path.isdir(x):
    return
  os.mkdir(x)
  if not os.path.isdir(x):
    raise RuntimeError("Failed to create dir %r" % x)

def save_checkpoint(state, is_best, directory):
  mkdir_if_not_exist(directory)

  filepath = os.path.join(directory, 'model.pth')
  torch.save(state, filepath)
  if is_best:
    best_acc1 = state['best_acc1'].item()
    best_filepath = os.path.join(directory, 'model_best_%5.3f.pth' % best_acc1)
    shutil.copyfile(filepath, best_filepath)
    print('Saving best ckpt to {}, acc1: {}'.format(best_filepath, best_acc1))
  return best_filepath if is_best else filepath

def evaluate(val_loader, model, criterion):
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

  return top1.avg

if __name__ == '__main__':

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device_ids = None if args.gpus == "" else [
      int(i) for i in args.gpus.split(",")
  ]

  model = resnet18()
  model.load_state_dict(torch.load(args.pretrained))

  traindir = os.path.join(args.data_dir, 'train')
  valdir = os.path.join(args.data_dir, 'validation')

  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  train_dataset = datasets.ImageFolder(
      traindir,
      transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
      ]))

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True)

  val_dataset = datasets.ImageFolder(
      valdir,
      transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
      ]))

  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)

  ana_dataset = torch.utils.data.Subset(val_dataset,
                                        list(range(args.ana_subset_len)))

  ana_loader = torch.utils.data.DataLoader(
      ana_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)
  gpus = get_gpus(args.gpus)
  model.to(device)
  criterion = torch.nn.CrossEntropyLoss().cuda()
  input_signature = torch.randn([1, 3, 224, 224], dtype=torch.float32)
  input_signature = input_signature.to(device)

  pruning_runner = get_pruning_runner(model, input_signature, args.method)

  if args.method == 'iterative':
    if args.load_slim_model:
      slim_model = pruning_runner.prune(
          removal_ratio=args.sparsity,
          mode='slim',
          channel_divisible=args.channel_divisible)
      ckpt = torch.load(args.slim_model)
      slim_model.load_state_dict(ckpt)
      print(evaluate(val_loader, model, criterion))
      model = slim_model
    else:
      pruning_runner.ana(eval_fn, args=(ana_loader,), gpus=gpus)
      sparse_model = pruning_runner.prune(
          removal_ratio=args.sparsity,
          mode='sparse',
          channel_divisible=args.channel_divisible)
      model = sparse_model

  elif args.method == 'one_step':
    if args.load_slim_model:
      slim_model = pruning_runner.prune(
          removal_ratio=args.sparsity,
          mode='slim',
          channel_divisible=args.channel_divisible)
      ckpt = torch.load(args.slim_model)
      slim_model.load_state_dict(ckpt)
      model = slim_model
      print(evaluate(val_loader, model, criterion))
    else:
      pruning_runner.search(
          gpus=gpus,
          calibration_fn=calibration_fn,
          eval_fn=eval_fn,
          num_subnet=args.num_subnet,
          removal_ratio=args.sparsity,
          calib_args=(train_loader,),
          eval_args=(ana_loader,))
      # index=None: select optimal subnet automatically
      slim_model = pruning_runner.prune(
          removal_ratio=args.sparsity,
          mode='slim',
          index=None,
          channel_divisible=args.channel_divisible)
      model = slim_model

  train(model, train_loader, val_loader, criterion, device_ids)

  # Generating the pruned model to do quantization
  from pytorch_nndct.utils import slim
  model = resnet18()
  slim_model = slim.load_state_dict(model,
                                    torch.load('resnet18_pruned_best.pth'))

  acc1 = evaluate(val_loader, slim_model, criterion)
  print(
      f'=> pruning ratio {args.sparsity} model completed. Top1 acc: {acc1:.4f}')

  # PTQ (Post Training Quantization)
  from pytorch_nndct.apis import torch_quantizer

  quantizer = torch_quantizer(
      args.quant_mode, slim_model, (input_signature), device=device)

  # if quant_mode is 'calib', ignore the accuracy log because it is not the final accuracy can be got.
  # only check the accuray if quant_mode is 'test'.
  acc1 = evaluate(val_loader, quantizer.quant_model, criterion)

  quantizer.export_quant_config()
  quantizer.export_torch_script()
  quantizer.export_onnx_model()

  # QAT (Quantization Aware Training)
  from pytorch_nndct import QatProcessor
  slim_model.train()
  qat_processor = QatProcessor(slim_model, input_signature)
  # qat step 1: get quantized model and train it.
  quantized_model = qat_processor.trainable_model()
  best_ckpt = train(quantized_model, train_loader, val_loader, criterion,
                    device_ids)
  # qat step 2: get deployable model and test it.
  # there may be some slight differences in accuracy with the quantized model.
  quantized_model.load_state_dict(torch.load(best_ckpt)['state_dict'])
  deployable_model = qat_processor.to_deployable(quantized_model, args.save_dir)

  # qat step 3: export xmodel from deployable model.
  deployable_model = qat_processor.deployable_model(
      args.save_dir, used_for_xmodel=True)
  val_subset = torch.utils.data.Subset(val_dataset, list(range(1)))
  subset_loader = torch.utils.data.DataLoader(
      val_subset,
      batch_size=1,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)
  # notice : must forward deployable model at least 1 iteration with batch_size = 1
  for images, _ in subset_loader:
    deployable_model(images)
  qat_processor.export_xmodel(args.save_dir)
