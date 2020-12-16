import os
import pytorch_nndct.nn as nndct_nn
import shutil
import time
import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import warnings

from pytorch_nndct.apis import torch_quantizer
from pytorch_nndct.nn.modules import functional

model_urls = {
    'resnet18': '/proj/rdi/staff/wluo/UNIT_TEST/models/resnet18.pth',
}

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
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
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
    #state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    state_dict = torch.load(model_urls[arch])
    model.load_state_dict(state_dict)
  return model

def resnet18(pretrained=False, progress=True, **kwargs):
  r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
  return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                 **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
  r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
  return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                 **kwargs)

def train(train_loader, model, criterion, optimizer, epoch, gpu=None):
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

    if gpu is not None:
      model = model.cuda(gpu)
      images = images.cuda(gpu, non_blocking=True)
    target = target.cuda(gpu, non_blocking=True)

    # compute output
    output = model(images)
    loss = criterion(output, target)

    l2_decay = 1e-4
    l2_norm = 0.0
    for name, param in model.named_parameters():
      if 'log_threshold' in name:
        l2_norm += torch.pow(param, 2.0)[0]
    loss += l2_decay * torch.sqrt(l2_norm)

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

def validate(val_loader, model, criterion, gpu):
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
      if gpu is not None:
        model = model.cuda(gpu)
        images = images.cuda(gpu, non_blocking=True)
      target = target.cuda(gpu, non_blocking=True)

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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, 'model_best.pth.tar')

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

def adjust_learning_rate(optimizer, epoch, lr, schedule=None):
  """Sets the learning rate to the initial LR decayed by schedule"""
  if schedule and epoch in schedule:
    lr *= 0.1
  else:
    lr = lr * (0.1**(epoch // 10))

  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

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
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
  #data = '/scratch/workspace/dataset/imagenet/raw-data'
  data='/proj/rdi/staff/niuxj/imagenet'
  workers = 4
  gpu = 0

  batch_size = 128
  lr = 1e-5
  momentum = 0.9
  weight_decay = 1e-4

  traindir = os.path.join(data, 'train')
  valdir = os.path.join(data, 'val')
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
      batch_size=batch_size,
      shuffle=True,
      num_workers=workers,
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
      batch_size=batch_size,
      shuffle=False,
      num_workers=workers,
      pin_memory=True)

  model = resnet18(pretrained=True)

  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss().cuda(gpu)

  input = torch.randn([batch_size, 3, 224, 224], dtype=torch.float32)
  quantizer = torch_quantizer(quant_mode = 'calib',
                              module = model,
                              input_args = input,
                              bitwidth = 8,
                              mix_bit = True,
                              qat_proc = True)
  quantized_model = quantizer.quant_model
  optimizer = torch.optim.Adam(
    quantized_model.parameters(), lr, weight_decay=weight_decay)
  print(quantized_model)

  # The test accuracy should be equal to the float model:
  # quantized_model.disable_quant()
  # validate(val_loader, quantized_model, criterion, gpu)
  # quantized_model.enable_quant()

  #quantized_model = torch.nn.DataParallel(quantized_model.cuda())
  best_acc1 = 0
  epochs = 2
  for epoch in range(epochs):
    adjust_learning_rate(optimizer, epoch, lr)

    # train for one epoch
    train(train_loader, quantized_model, criterion, optimizer, epoch, gpu)

    # evaluate on validation set
    acc1 = validate(val_loader, quantized_model, criterion, gpu)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': quantized_model.state_dict(),
        'best_acc1': best_acc1}, is_best)
    print('Saving ckpt with best_acc1:', best_acc1)

  #quantized_model.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])
  #deployable_model = qat_sched.convert_to_deployable(quantized_model, mix_bit=True)
  quantizer.deploy(quantized_model, mix_bit=True)
  deployable_model = quantizer.deploy_model
  deployed_acc1 = validate(val_loader, deployable_model, criterion, gpu)

  quantized_model.freeze_bn()
  quantized_acc1 = validate(val_loader, quantized_model, criterion, gpu)

  if quantized_acc1 != deployed_acc1:
    warnings.warn(
        'The accuracy of deployed model is not equal to the accuracy of quantized model.')

  val_dataset2 = torch.utils.data.Subset(val_dataset, list(range(1)))
  val_loader2 = torch.utils.data.DataLoader(
      val_dataset2,
      batch_size=1,
      shuffle=False,
      num_workers=workers,
      pin_memory=True)
  validate(val_loader2, deployable_model, criterion, gpu)
  quantizer.export_xmodel()

if __name__ == '__main__':
  main()
