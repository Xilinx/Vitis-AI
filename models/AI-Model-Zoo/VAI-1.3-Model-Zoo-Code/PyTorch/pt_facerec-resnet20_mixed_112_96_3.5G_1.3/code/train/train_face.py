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

# PART OF THIS FILE AT ALL TIMES.

import argparse
import pdb
import os
import shutil
import time
import math
import copy
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import numpy as np

import sys
sys.path.append('../models/')
sys.path.append('../utils/')
from load_imglist import ImageList
import face_model
import evaluate
import image_transforms
import Log
from torchsummary import summary

parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_am',
                    help='model architecture: (default: vgg19)')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epoches', default=30, type=int, metavar='N',
                    help='number of total epoches to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--test-batch-size', default=128, type=int,
                    metavar='N', help='test mini-batch size (default: 500)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', '--learning-rate-decay', default=0.1, type=float,
                    metavar='LRD', help='learning rate decay rate')
parser.add_argument('--lr-list', nargs='+', default=[13,20,25], type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--model', default='', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29, LightCNN-29v2')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--finetune', default=True, action='store_true', help='use tensorboard.')
parser.add_argument('--root_path', default='../../data/train/ms_glint', 
                    type=str, metavar='PATH', help='path to root path of images (default: none)')
parser.add_argument('--train_list', default='../../data/train/ms_glint/list/msra_celebrity.txt', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--testset_images', default='../../data/test/lfw/images', type=str, metavar='PATH',
                    help='path to testset (default: none)')
parser.add_argument('--testset_pairs', default='../../data/test/lfw/pairs.txt', type=str, metavar='PATH',
                    help='path to testset (default: none)')
parser.add_argument('--testset_list', default='../../data/test/lfw/lfw.txt', type=str, metavar='PATH',
                    help='path to testset (default: none)')
parser.add_argument('--save-dir', default='save', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--num_classes', default=180855, type=int,
                    metavar='N', help='number of classes')
parser.add_argument('--save-prec', default=90, type=float, metavar='S',
                    help='save prec')
parser.add_argument('--test-epoch', default=0, type=int, metavar='S',
                    help='test epoch')
parser.add_argument('--fea-dim', default=512, type=int)
parser.add_argument('--prefix', default='temp', type=str, metavar='PREFIX',
                    help='checkpoint prefix (default: none)')
parser.add_argument('--summary', action='store_true', help='use tensorboard.')
parser.add_argument('--lamb', default=0.001, type=float, help='lambda for lamsoftmax')
parser.add_argument('--scale', default=80, type=float, help='scale for amsoftmax')
parser.add_argument('--margin', default=0.4, type=float, help='margin for amsoftmax or lamsoftmax')
parser.add_argument('--eval', default='lfw', type=str)
parser.add_argument('--auggamma', default='1.,1.', type=str)
parser.add_argument('--resize', default='', type=str)
parser.add_argument('--nonlinear', default='relu', type=str)
parser.add_argument('--log', default='log', type=str)

args = parser.parse_args()

Log.Log(args.log, 'w+', 1) # set log file

print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

# tensorboard
def to_np(x):
    return x.data.cpu().numpy()

if args.summary:
    from logger import Logger
    log_dir = os.path.join('./runs/', args.prefix)
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    logger = Logger(log_dir)

start = time.time()

def main():
    global args, best_prec1, logger
    best_prec1 = 0.

    # create resnet for face recognition
    if 'resnet' in args.arch:
        truearch = args.arch.replace('_am', '').replace('_arc', '')
        if 'am' in args.arch or 'arc' in args.arch:
            ring = {'s': args.scale}
            model = face_model.__dict__[truearch](args.num_classes, wn=True, fn=True, nonlinear=args.nonlinear, fea_dim=args.fea_dim, ring=ring)
        else:
            model = face_modle.__dict__[truearch](args.num_classes, wn=False, fn=False,nonlinear=args.nonlinear, fea_dim=args.fea_dim)
    size = (3, 112, 96)
    summary(copy.deepcopy(model.cuda()), size)

    model = torch.nn.DataParallel(model).cuda()
    if args.finetune:
        checkpoint = args.pretrained
        if os.path.isfile(checkpoint):
            checkpoint = torch.load(checkpoint)
            # load part of state_dict
            pretrained_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            new_dict = {}
            for k, v in pretrained_dict.items():
                if not k in model_dict:
                    continue
                else:
                    new_dict[k] = v
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(new_dict)
            model.load_state_dict(model_dict)
        else:
            print("wrong checkpoint path", checkpoint)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    grayscale = transforms.Grayscale(num_output_channels=1)
    
    auggamma = [float(x) for x in args.auggamma.split(',')]
    randomgamma = image_transforms.RandomGamma(auggamma[0], auggamma[1])
    gammabalance = image_transforms.GammaBalance()
    transform_ops = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ]
    if args.resize != '':
        size = (int(x) for x in args.size.split(','))
        resize = transforms.Resize(size=size)
        transform_ops.insert(1, resize)
    transform=transforms.Compose(transform_ops)
    #load image
    train_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, fileList=args.train_list, transform=transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()
    
    eval_best_prec = 0.
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epoches):

        if epoch in args.lr_list:
            optimizer.param_groups[0]['lr'] *= args.lr_decay
        # adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        prec1 = train(train_loader, model, criterion, optimizer, epoch)

        modelname = os.path.join(args.save_dir, 'checkpoint_{}_{}.tar'.format(args.prefix, epoch+1))

        # remember best prec@1 and save checkpoint
        if epoch >= args.test_epoch:
            best_prec1 = prec1
            suffix = '_{}.bin'.format(args.arch)

            eval_prec = evaluate.eval_lfw(model, args.testset_images, args.testset_list, args.testset_pairs, batch_size=args.test_batch_size, fea_dim=args.fea_dim)
            if eval_prec > eval_best_prec:
                eval_best_prec = eval_prec
                best_epoch = epoch + 1

                save_checkpoint({
                    'epoch': best_epoch,
                    'lr': optimizer.param_groups[0]['lr'], 
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    }, filename=modelname)
                print('Eval {}\tprec: {}\tbest_prec: {}'.format(args.eval, eval_prec, eval_best_prec))

        best_prec1 = max(prec1, best_prec1)
        print('{} best prec: {}, method: {}, best epoch: {}'.format(args.eval, eval_best_prec, args.prefix, best_epoch))

def train(train_loader, model, criterion, optimizer, epoch):
    losses     = AverageMeter()
    mean_cos   = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    model.train()

    for i, (input, target) in enumerate(train_loader):
        if args.cuda:
            input      = input.cuda()
            target     = target.cuda()
        input_var  = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        ringloss = torch.autograd.Variable(torch.Tensor([0]), requires_grad=True)
        feature, output = model(input_var)
        
        if '_am' in args.arch:
            cos = am_margin(output, target_var, args.margin)
            output = model.module.scale(output)
            mean_cos.update(cos, 1) 
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data, input.size(0)) # loss.data[0] 
        top1.update(prec1, input.size(0)) # prec1[0]
        top5.update(prec5, input.size(0)) # prec5[0]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            print('Epoch: [{0}][{1}/{2}] '
                    'Algo {algo}  '
                    'Time {t:.1f} '
                    'Lr {lr:.5f} '
                  'Loss {loss.val:.1f} ({loss.avg:.1f}) '
                  'Cos {mean_cos.val:.2f} '
                  'Prec@1 {top1.avg:.3f}'.format(
                          epoch, i, len(train_loader), algo=args.prefix[:-5], t=time.time()-start, lr=lr,
                   mean_cos=mean_cos, loss=losses, top1=top1))

        if args.summary:
            iters = epoch * len(train_loader) + i
            logger.scalar_summary('train_loss', float(losses.avg), iters)
            logger.scalar_summary('train_acc', float(top1.avg), iters)

    return top1.avg

def am_margin(output, target, m):
    t = to_np(copy.copy(target))
    n = output.size()[0]
    n = list(range(n))
    mean_cos = torch.mean(output[n, t]).item() # target logit
    output[n, t] = output[n, t] - m
    return mean_cos
    

# little different with mxnet version
# https://github.com/deepinsight/insightface/blob/1436c181056c703029d67dbadeb1408ffa14c59d/src/train_softmax.py#L228
def arc_margin(output, target, m):
    t = to_np(target)
    n = output.size()[0]
    n = list(range(n))
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mean_cos = torch.mean(output[n, t]).item() # target logit
    cos_t = 1 * output[n, t]
    sin_t = torch.sqrt(1 - cos_t * cos_t)
    output[n, t] = cos_m * cos_t - sin_m * sin_t
    return mean_cos

def save_checkpoint(state, filename):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    scale = 0.457305051927326
    step  = 10
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
