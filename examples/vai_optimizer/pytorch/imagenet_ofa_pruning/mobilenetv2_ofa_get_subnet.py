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
import torchvision.datasets as datasets
from torchvision.models.mobilenet import mobilenet_v2
import torchvision.transforms as transforms
from pytorch_nndct import OFAPruner

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
# ofa config
parser.add_argument(
    '--channel_divisible', type=int, default=8, help='make channel divisible')
parser.add_argument(
    '--expand_ratio', type=list, default=[0.25, 0.5, 0.75, 1], help='expand ratio')
parser.add_argument(
    '--excludes',
    type=list,
    default=None,
    help='excludes module')

parser.add_argument(
    '--targeted_macs',
    type=int,
    default=240,
    help='targeted macs subnet need to be export')

parser.add_argument(
    '--pretrained_ofa_model',
    type=str,
    default='model_best.pth.tar',
    help='Pretrained ofa model filepath')
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

def evaluate(dataloader, model, criterion):
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

  return float(top1.avg), float(top5.avg)

if __name__ == '__main__':

  gpus = get_gpus(args.gpus)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = mobilenet_v2(pretrained=True)

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

  model.to(device)

  input_signature = torch.randn([1, 3, 224, 224], dtype=torch.float32).cuda()

  ofa_pruner = OFAPruner(model, input_signature)
  ofa_model = ofa_pruner.ofa_model(args.expand_ratio, args.channel_divisible,
                                   args.excludes)

  model = ofa_model.cuda()

  #reloading model
  with open(args.pretrained_ofa_model, 'rb') as f:
    checkpoint = torch.load(f, map_location='cpu')
  assert isinstance(checkpoint, dict)
  pretrained_state_dicts = checkpoint['state_dict']
  for k, v in model.state_dict().items():
    name = 'module.' + k if not k.startswith('module') else k
    v.copy_(pretrained_state_dicts[name])

  pareto_global = ofa_pruner.load_subnet_config('pareto_global.txt')

  static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
      ofa_model, pareto_global[str(args.targeted_macs)]['subnet_setting'])

  model = static_subnet

  model.cuda()

  with torch.no_grad():
    model.eval()
    ofa_pruner.reset_bn_running_stats_for_calibration(model)
    for batch_idx, (images, _) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      model(images)
      if batch_idx >= 16:
        break

  criterion = torch.nn.CrossEntropyLoss().cuda()

  top1, top5 = evaluate(val_loader, model, criterion)

  print('subnet Acc@top1:', top1)
  print('subnet macs:', macs)
  print('subnet params:', params)
