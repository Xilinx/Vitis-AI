import argparse
import shutil
import time

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn

from pytorch_nndct.nn.modules import functional

torch.backends.cudnn.deterministic = True

torch.cuda.set_device(0)

'''
  Note:
    1.MobileNetV2: change the global avgpooling func to the ResNet format
  Result:
    ReLU6
    baseline: FP32module: top1: 71.870 top5: 90.294
              FP32module: top1: 71.870 top5: 90.294   Conv + BN -> fold
              FP32module: top1: 71.434 top5: 90.142   Conv + BN -> fold  + CLE(Cross Layer Equalization)
    ---without CLE
    top1: 71.576  top5: 90.190  BFP(bfp_bitwidth: 16,"bfp_tile_size": 8, mantissa: 7  (16 - 1(sign) - 8(exp))
    top1: 68.810  top5: 88.512  BFP(bfp_bitwidth: 15,"bfp_tile_size": 8, mantissa: 6  (15 - 1(sign) - 8(exp))
    top1: 66.996  top5: 87.290  BFP(bfp_bitwidth: 14,"bfp_tile_size": 8, mantissa: 5  (14 - 1(sign) - 8(exp))
    top1: 58.298  top5: 80.948  BFP(bfp_bitwidth: 13,"bfp_tile_size": 8, mantissa: 4  (13 - 1(sign) - 8(exp)) 
    ---with CLE
    top1: 71.434  top5: 90.142  BFP(bfp_bitwidth: 16,"bfp_tile_size": 8, mantissa: 7  (16 - 1(sign) - 8(exp)) conv + BN = fold+ CLE。
    top1: 70.878  top5: 89.664  BFP(bfp_bitwidth: 15,"bfp_tile_size": 8, mantissa: 6  (15 - 1(sign) - 8(exp)) conv + BN = fold+ CLE。
    top1: 69.320  top5: 88.998  BFP(bfp_bitwidth: 14,"bfp_tile_size": 8, mantissa: 5  (14 - 1(sign) - 8(exp)) conv + BN = fold+ CLE。
    top1: 53.890  top5: 77.678  BFP(bfp_bitwidth: 13,"bfp_tile_size": 8, mantissa: 4  (13 - 1(sign) - 8(exp))  conv + BN = fold+ CLE。
  Prepare using:
    nndct_shared/optimization/commander.py -->modify the _ACT_TYPES to: [NNDCT_OP.RELU, NNDCT_OP.RELU6]
  Using(for internal use): 
    python MobileNetV2.py --config_file config.json
'''

__all__ = ['MobileNetV2'] 

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', default='MobileNetV2', type=str)
parser.add_argument(
    '--config_file',
    default='config.json',
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
parser.add_argument('--batch_size', default=100, type=int, help='batch_size')
parser.add_argument(
    '--val_freq_iters',
    default=80000,
    type=int,
    help='Run validation every n iters during training.')
parser.add_argument('--num', default=10, type=int, help='count')
parser.add_argument('--optim', default='AdamW', type=str, help='SGD or AdamW')
parser.add_argument('--use_pre', default=0, type=int, help='')
parser.add_argument('--use_best_cur', default=0, type=int, help='')
parser.add_argument(
    '--pretrained',
    default="/group/dphi_algo_scratch_08/zijunx/data/mobilenetv2.pth",
    type=str,
    help='Path of pretrained weights')
parser.add_argument('--acc', default=71.870, type=int, help='fp32, \
    71.870 for all act func ReLU6 and 71.840 for all act func ReLU')
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


# MobileNet V2
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.add = functional.Add()

    def forward(self, x):
        if self.use_res_connect:
            return self.add(x, self.conv(x))
            # return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        # avgpooling 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Identity(), # modify the Dropout  to Identity
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = self.avgpool(x).reshape(x.shape[0], -1)
        # x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

def load_imagenet(batch_size):
  # Data loading code
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  # print(torchvision.get_image_backend()) //PIL
  train_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(
          '/group/dphi_gpu_scratch/yuwang/dataset/imagenet/pytorch/train',
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
          '/group/dphi_gpu_scratch/yuwang/dataset/imagenet/pytorch/val',
          transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              normalize,
          ])),
      batch_size=batch_size,
      shuffle=False,
      num_workers=32,
      pin_memory=True)

  return train_loader, val_loader

def train(train_loader, val_loader, model, criterion, optimizer, epoch, args,
          best_acc1):
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

    images = images.cuda()
    target = target.cuda()

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
      acc1 = validate(val_loader, model, criterion, args)

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

    if i == args.num * args.val_freq_iters:
      break

  return best_acc1

def adjust_learning_rate(optimizer, epoch, args):
  """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
  lr = args.lr * (0.1**(epoch // 10))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def validate(val_loader, model, criterion, args):

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
      #if i == 1:
      #  break

      images = images.cuda()
      target = target.cuda()

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

    # TODO: this should also be done with the ProgressMeter

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

  # prepare the official torch version mobilenet v2 (without some modify in MobileNetV2)
  model = MobileNetV2().cuda()
  args.pretrained = '/group/dphi_algo_scratch_08/zijunx/data/mobilenetv2.pth'
  args.acc = 71.870  

  # initialize parameters
  state_dict = torch.load(args.pretrained)
  model.load_state_dict(state_dict, strict=True) # as you del the Dropout need to change to strict =true

  # data loading
  train_loader, val_loader = load_imagenet(args.batch_size)
  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss().cuda()

  from pytorch_nndct.quantization import bfp
  inputs = torch.randn([args.batch_size, 3, 224, 224],
                       dtype=torch.float32).cuda()
  model = bfp.quantize_model(model, inputs, args.config_file)

  #from pytorch_nndct.apis import torch_quantizer
  #from pytorch_nndct.utils import module_util
  #from nndct_shared.utils import set_option_value
  #set_option_value("nndct_quant_off", True)
  #quantizer = torch_quantizer('calib', model, inputs)
  #model = quantizer.quant_model
  #module_util.enable_print_blob(model)

  best_acc1 = validate(val_loader, model, criterion, args)
  print("model: %s" % args.model, flush=True)

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
                        epoch, args, best_acc1)

  if args.epochs > 0:
    validate(val_loader, model, criterion, args)
