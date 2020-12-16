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

from data import *
from data.det import *
from data.seg import *
from utils.det_augmentations import DetAugmentation
from utils.seg_augmentations import SegAugmentation
from layers.modules import MultiBoxLoss, MultiBoxOriLoss
import model_res18
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from loss import *
from collections import OrderedDict


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./float/train',
                    help='Directory for saving checkpoint models')
parser.add_argument('--DET_ROOT',default='/scratch/workspace/data/multi_task_det5_seg16/detection/Waymo_bdd_txt',
                    help='Directory for detection data')
parser.add_argument('--SEG_ROOT',default='/scratch/workspace/data/multi_task_det5_seg16/segmentation',
                    help='Directory for segmentation data')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    print('Start training')
    dataset_seg = Segmentation(SEG_ROOT=args.SEG_ROOT,
                                  transform=SegAugmentation(solver['resize'],
                                                         MEANS))
    dataset_det = Detection(DET_ROOT=args.DET_ROOT,
                               transform=DetAugmentation(solver['resize'],
                                                         MEANS))

    print('dataset create success')
    
    net = model_res18.build_model(solver['det_classes'], solver['seg_classes'])
    
    print('build model success')
   
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        state_dict = torch.load(args.resume)
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            if k[0:7] == 'module.':
                name = k[7:]
                new_state_dict[name] = v
            else:
                name = k
                new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
 
    
    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.cuda:
        net = net.cuda()

    # step_index = 0
    # start_lr = args.lr
    step_index = sum([args.start_iter > lr_step for lr_step in solver['lr_steps']])
    start_lr = args.lr * (args.gamma ** (step_index))

    optimizer = optim.SGD(net.parameters(), lr=start_lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(solver['det_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    
    net.train()
    epoch_size_det = len(dataset_det) // args.batch_size
    epoch_size_seg = len(dataset_seg) // args.batch_size
    
    print('Training SSD on: Cityscapes and BDD-100k')
    print('Using the specified args:')
    print(args)

    data_loader_seg = data.DataLoader(dataset_seg, int(args.batch_size),
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=segmentation_collate,
                                  pin_memory=True)

    data_loader_det = data.DataLoader(dataset_det, int(args.batch_size),
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)


    # create batch iterator
    batch_iterator_seg = iter(data_loader_seg)
    batch_iterator_det = iter(data_loader_det)
    for iteration in range(args.start_iter, solver['max_iter']):
        if iteration != 0 and (iteration % epoch_size_seg == 0):
            batch_iterator_seg = iter(data_loader_seg)
        
        if iteration != 0 and (iteration % epoch_size_det == 0):
            batch_iterator_det = iter(data_loader_det)

        if iteration in solver['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
        
        optimizer.zero_grad()
        for iter_round in range(1):
            # load train data
            images_seg, seg = next(batch_iterator_seg)
            images_det, targets = next(batch_iterator_det)

            if args.cuda:
                images_seg = Variable(images_seg.cuda())
                images_det = Variable(images_det.cuda())
                with torch.no_grad():
                    targets = [Variable(ann.cuda()) for ann in targets]
                    seg = Variable(seg.cuda())
            else:
                images_seg = Variable(images_seg)
                images_det = Variable(images_det)
                with torch.no_grad():
                    targets = [Variable(ann) for ann in targets]
                    seg = Variable(seg)
            # forward
            t0 = time.time()
            out_seg = net(images_seg)
            # backprop
            _, _, seg_data, _ = out_seg
            loss_m = cross_entropy2d(seg_data, seg)
            loss_m.backward() 
            
            out_det = net(images_det)
            loss_l, loss_c = criterion(out_det, targets)
            loss = loss_l + loss_c 
            loss.backward()


        optimizer.step()
        t1 = time.time()

        # if iteration % 10 == 0:
        if iteration % 50 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration), end=' ')
            print('Learning_Rate: %f ||' % (optimizer.param_groups[0]['lr']), end=' ')
            print('Conf_Loss: %.4f ||' % (loss_c.data), end=' ')
            print('Loc_Loss: %.4f ||' % (loss_l.data), end=' ')
            print('Seg_Loss: %.4f ||' % (loss_m.data), end=' ')

        if iteration % 500 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), os.path.join(args.save_folder, 'iter_' +
                                                      repr(iteration) + '.pth'))
    
    torch.save(net.state_dict(),
               args.save_folder + 'final' + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    train()

