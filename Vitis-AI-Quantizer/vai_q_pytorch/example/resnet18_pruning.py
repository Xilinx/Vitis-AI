import os
import argparse
import random
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
from tqdm import tqdm

from pytorch_nndct.nndct.pruning import pruning

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default="/scratch/workspace/wangyu/nndct_test_data/imagenet/data",
    help='Data set directory')
parser.add_argument(
    '--model_dir',
    default="/scratch/workspace/wangyu/nndct_test_data/models",
    help='Trained model file path')
args, _ = parser.parse_known_args()

def load_data(train=True,
              data_dir='dataset/imagenet',
              batch_size=128,
              subset_len=None,
              sample_method='random',
              distributed=False,
              model_name='resnet18',
              **kwargs):
  # random.seed(12345)
  traindir = data_dir + '/train'
  valdir = data_dir + '/val'
  train_sampler = None
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  if model_name == 'inception_v3':
    size = 299
    resize = 299
  else:
    size = 224
    resize = 256
  if train:
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    if distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **kwargs)
  else:
    dataset = torchvision.datasets.ImageFolder(valdir,)
    dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]))
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, **kwargs)
  return data_loader, train_sampler

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
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def evaluate(model, val_loader, loss_fn):

  model.eval()
  model = model.cuda()
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  total = 0
  Loss = 0
  for iteration, (images, labels) in tqdm(
      enumerate(val_loader), total=len(val_loader)):
    images = images.cuda()
    labels = labels.cuda()
    #pdb.set_trace()
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    Loss += loss.item()
    total += images.size(0)
    acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
    top1.update(acc1[0], images.size(0))
    top5.update(acc5[0], images.size(0))
  return top1.avg, top5.avg, Loss / total

def ana_eval_fn(model, val_loader, loss_fn):
  return evaluate(model, val_loader, loss_fn)[1]

if __name__ == '__main__':
  model_name = 'resnet18'
  file_path = os.path.join(args.model_dir, model_name + '.pth')
  model = resnet18().cpu()
  model.load_state_dict(torch.load(file_path))

  # to get loss value after evaluation
  loss_fn = torch.nn.CrossEntropyLoss().cuda()

  batch_size = 100
  val_loader, _ = load_data(
      train=False,
      data_dir=args.data_dir,
      batch_size=batch_size,
      subset_len=2000,
      sample_method='random',
      model_name=model_name)

  pruner = pruning.Pruner(model, (3, 224, 224))
  #pruner.ana(ana_eval_fn, args=(val_loader, loss_fn))
  pruned_model = pruner.prune(threshold=0.001)

  acc1, acc5, loss = evaluate(model, val_loader, loss_fn)
  print("Baseline model: acc1 = {}, acc5 = {}, loss = {}".format(
      acc1, acc5, loss))

  acc1, acc5, loss = evaluate(pruned_model, val_loader, loss_fn)
  print("Pruned model: acc1 = {}, acc5: {}, loss = {}".format(acc1, acc5, loss))
