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



import os
import random
import torch
import numpy as np
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils

from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F

from code.data_loader import cityscapes
from code.models.enet_xilinx import ENet
from code.utils import evaluate, AverageMeter, CrossEntropyLoss2d, save_checkpoint, compute_flops
cudnn.benchmark = True
import torch.nn as nn
from collections import OrderedDict
import tqdm
from code.configs.model_config import Configs

def main(args):

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)  
    open(os.path.join(args.checkpoint_dir, 'arguments.txt'), 'w').write(str(args) + '\n\n')

    # network
    net = ENet(num_classes=args.num_classes)
    # flops 
    current_epoch =0 

    if args.resume:
        checkpoint = torch.load(args.weight, map_location='cuda:0')
        # strict=False, so that it is compatible with old pytorch saved models
        checkpoint['state_dict'] = OrderedDict([(k[5:], v) if 'base' in k else (k, v) for k, v in checkpoint['state_dict'].items()])
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))
        current_epoch = checkpoint['epoch']
    else:
        print("=> training from scratch!")
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    
    train_set = cityscapes.CityScapes(root=args.data_root, quality='fine', mode='train', size=(1024, 512))
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=16, shuffle=True)
    
    val_set = cityscapes.CityScapes(root=args.data_root, quality='fine', mode='val', size=(1024, 512))
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, num_workers=16, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],'lr': 2 * args.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'], 'lr': args.lr, 
         'weight_decay': args.weight_decay}], momentum=args.momentum)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, min_lr=1e-10)
    best_miou = 0.0

    if args.test_only:
        _ = validate(val_loader, net, criterion, optimizer, current_epoch, args, best_miou)    
        exit()
    else:
        for epoch in range(current_epoch, args.total_epoch):
            train(train_loader, net, criterion, optimizer, epoch, args)
            val_loss, best_miou = validate(val_loader, net, criterion, optimizer, epoch, args, best_miou)
            scheduler.step(val_loss)

def train(train_loader, net, criterion, optimizer, epoch, args):

    net.train()
    train_loss = AverageMeter()
    curr_iter = epoch* len(train_loader)
    max_iters = args.total_epoch * len(train_loader)
    for i, data in enumerate(train_loader):
        optimizer.param_groups[0]['lr'] = 2 * args.lr *(1 -  float(curr_iter) / max_iters) ** args.lr_decay
        optimizer.param_groups[1]['lr'] = args.lr *(1 -  float(curr_iter) / max_iters) ** args.lr_decay
        inputs, labels = data
        N = inputs.size(0)
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        optimizer.zero_grad()
        outputs = net(inputs)
        if outputs.size()[2:] != labels.size()[1:]:
            outputs = F.upsample(outputs, size=gts.size()[1:], mode='bilinear')
        loss = criterion(outputs, labels) 
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data)

        curr_iter += 1

        if (i + 1) % args.print_freq == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [lr %.10f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg, optimizer.param_groups[1]['lr']))


def validate(val_loader, net, criterion, optimizer, epoch, args, best_miou):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        inputs, gts = data
        with torch.no_grad():
            inputs = inputs.cuda()
            gts = gts.cuda()
            outputs = net(inputs)
            if outputs.size()[2:] != gts.size()[1:]:
                outputs = nn.functional.interpolate(outputs, size=gts.size()[1:], mode='bilinear', align_corners=True)
            predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
            val_loss.update(criterion(outputs, gts).data)
    
            inputs_all.append(inputs.data.cpu())
            gts_all.append(gts.data.cpu().numpy())
            predictions_all.append(predictions)

    gts_all = np.concatenate(gts_all)
    predictions_all = np.concatenate(predictions_all)

    acc, acc_cls, ious, mean_iu, fwavacc = evaluate(predictions_all, gts_all, args.num_classes, args.ignore_label)
    print('[epoch %d], [val loss %.5f], [mean_iu %.5f]' % (
        epoch, val_loss.avg, mean_iu))

    if mean_iu > best_miou:
        best_miou = mean_iu
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_miou,
        }, args.checkpoint_dir, True)
        print('-------------------------------------------------------------')
        print('best record: [val loss %.5f], [mean_iu %.5f], [epoch %d]' % (
            val_loss.avg, mean_iu, epoch))
        print('-------------------------------------------------------------')

    return val_loss.avg, best_miou


if __name__ == '__main__':
    args = Configs().parse()
    torch.manual_seed(args.seed)
    main(args)

