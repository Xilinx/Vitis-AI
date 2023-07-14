import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pytorch_nndct import get_pruning_runner

parser = argparse.ArgumentParser()
parser.add_argument(
    '--method', type=str, default='iterative', help='iterative/one_step')
parser.add_argument(
    '--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument(
    '--sparsity', type=float, default=0.5, help='Sparsity ratio')
parser.add_argument(
    '--ana_subset_len',
    type=int,
    default=2000,
    help='Subset length for evaluating model in analysis, using the whole validation dataset if it is not set'
)
parser.add_argument(
    '--num_subnet', type=int, default=20, help='Total number of subnet')
parser.add_argument('--epoches', type=int, default=1, help='Train epoch')
parser.add_argument(
    '--pretrained',
    type=str,
    default='mynet.pth',
    help='Pretrained model filepath')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./cifar10',
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

def train(train_loader, model, criterion, optimizer, epoch):
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

    # compute output
    output = model(images)
    loss = criterion(output, target)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), images.size(0))
    top1.update(acc1[0], images.size(0))
    top5.update(acc5[0], images.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % 10 == 0:
      progress.display(i)

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

  return top1.avg, top5.avg

def load_weights(model, model_path):
  checkpoint = torch.load(model_path)
  model.load_state_dict(checkpoint)
  return model

def get_dataloader():
  transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

  val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

  ana_dataset = torch.utils.data.Subset(val_dataset,
                                        list(range(args.ana_subset_len)))

  ana_loader = torch.utils.data.DataLoader(
      ana_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)
  return train_loader, val_loader, ana_loader

if __name__ == '__main__':
  model = MyNet()
  model.cuda()
  train_loader, val_loader, ana_loader = get_dataloader()
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
  input_signature = torch.randn([1, 3, 32, 32], dtype=torch.float32)
  input_signature = input_signature.cuda()

  pruning_runner = get_pruning_runner(model, input_signature, args.method)

  if args.method == 'iterative':
    pruning_runner.ana(eval_fn, args=(val_loader,), gpus=['0'])
    model = pruning_runner.prune(removal_ratio=args.sparsity, mode='sparse')

  elif args.method == 'one_step':
    pruning_runner.search(
        gpus=['0'],
        calibration_fn=calibration_fn,
        calib_args=(val_loader,),
        eval_fn=eval_fn,
        eval_args=(val_loader,),
        num_subnet=args.num_subnet,
        removal_ratio=args.sparsity)
    # index=None: select optimal subnet automatically
    model = pruning_runner.prune(removal_ratio=args.sparsity, index=None, mode='slim')

  optimizer = torch.optim.Adam(
      model.parameters(), args.lr, weight_decay=args.weight_decay)

  best_acc1 = 0
  for epoch in range(args.epoches):
    train(train_loader, model, criterion, optimizer, epoch)
    acc1, acc5 = evaluate(val_loader, model, criterion)
    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if is_best:
      torch.save(model.state_dict(), 'mynet_pruned.pth')

