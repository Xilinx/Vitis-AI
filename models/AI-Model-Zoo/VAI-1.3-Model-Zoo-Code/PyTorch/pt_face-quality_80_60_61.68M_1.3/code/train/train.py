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

#-*-coding:utf-8-*-
import os
import sys
if os.environ["W_QUANT"]=='1':
    # load quant apis
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import numpy as np
sys.path.append('../models/')
sys.path.append('../utils/')
sys.path.append('../../data/')
sys.path.append('../configs/')

from model_config import Configs

import data
from torchsummary import summary
import model
import Log

def train_net(args, net, train_loader, test_loader):
    # set net on gpu
    net.to(device)
    summary(net, (3, args.size[0], args.size[1]))
    # loss and optimizer
    # points_criterion = nn.MSELoss(size_average=False)
    # points_criterion = nn.SmoothL1Loss(size_average=False)
    points_criterion = nn.L1Loss(size_average=False)
    quality_criterion = nn.L1Loss(size_average=False)
    optimizer = optim.SGD(net.parameters(), lr = args.base_lr, momentum = args.momentum, weight_decay = args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = args.milestones, gamma = args.gamma)
    # initial test
    min_error_p, min_error_q = eval_net(net, test_loader, 0)
    best_epoch_p, best_epoch_q = 0, 0
    # epochs
    for epoch in range(args.epochs):
        # train
        print('TRAIN[{}] {} {}'.format(epoch+1, time.strftime('%y-%m-%d %H:%M:%S', time.localtime()), name[:-10]))
        net.train()
        scheduler.step()
        for i, (images, labels) in enumerate(train_loader):
            net.zero_grad()
            images = images.to(device)
            points_labels, quality_labels, flags = labels
            flags = flags.data.numpy()
            points_labels = points_labels.to(device)
            quality_labels = quality_labels.to(device)
            points_outputs, quality_outputs = net(images)
            # normalize
            # points_outputs[:5] = points_outputs[:5] * 72
            # points_outputs[5:] = points_outputs[5:] * 96
            # 80x60
            points_outputs = points_outputs * 6. / 5.

            points = np.where(flags == 0)[0]
            quality = np.where(flags == 1)[0]
            points_outputs[quality] = 0. * points_outputs[quality]
            quality_outputs[points] = 0. * quality_outputs[points]
            # points loss
            len_p = max(len(points), 1)
            len_q = max(len(quality), 1)
            points_loss = args.pointsNum * points_criterion(points_outputs, points_labels) / (10. * len_p)
            # quality loss
            quality_loss = quality_criterion(quality_outputs, quality_labels) / (1. * len_q)
            
            loss = points_loss + quality_loss
            loss.backward()
            optimizer.step()
            if i % args.print_freq == 0:
                print('epoch {epoch:3d}, {i:3d}|{len:3d}, points_loss: {points_loss:2.4f}, ' \
                        'quality_loss: {quality_loss:2.4f}'.format(epoch=epoch+1, i=i, len=len(train_loader), \
                                points_loss=points_loss.item(), quality_loss=quality_loss.item()))
            # if i % 200 == 0:
                # error_p, error_q = eval_net(net, test_loader, epoch + 1)
                # net.train()
        error_p, error_q = eval_net(net, test_loader, epoch + 1)
        if error_p < min_error_p:
            min_error_p = error_p
            best_epoch_p = epoch + 1
            torch.save(net.state_dict(), 'save/{}.pth'.format(name))
        if error_q < min_error_q:
            min_error_q = error_q
            best_epoch_q = epoch + 1
        print('TEST points: error: {:.4} min_error: {:.4} best_epoch: {}'.format(error_p, min_error_p, best_epoch_p))
        print('    quality: error: {:.4} min_error: {:.4} best_epoch: {}'.format(error_q, min_error_q, best_epoch_q))

def eval_net(net, test_loader, epoch=0):
    # set net on gpu
    net.to(device)
    net.eval()
    points_distance = 0.
    quality_distance = 0.
    points_total = 0
    quality_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            points_outputs, quality_outputs = net(images)
            points_labels, quality_labels, flags = labels
            points_labels = points_labels.to(device)
            quality_labels = quality_labels.to(device)

            flags = flags.data.numpy()
            points = np.where(flags == 0)[0]
            quality = np.where(flags == 1)[0]
            points_outputs[quality] = 0. * points_outputs[quality]
            quality_outputs[points] = 0. * quality_outputs[points]

            # points
            points_outputs = points_outputs.to(device)
            quality_outputs = quality_outputs.to(device)
            # 80x60
            points_outputs = points_outputs * 6./5.
            # normalize
            # points_labels[:5] = points_labels[:5] * 72
            # points_outputs[5:] = points_outputs[5:] * 96
            # quality_outputs = 300. * quality_outputs
            # quality_labels = 300. * quality_labels
            points_distance += torch.sum(torch.abs(points_outputs - points_labels)).item()
            points_total += len(points)
            quality_distance += torch.sum(torch.abs(quality_outputs -quality_labels)).item()
            quality_total += len(quality)

    print(points_total, quality_total)
    points_total = max(points_total, 1)
    quality_total = max(quality_total, 1)
    return points_distance / (10*points_total), quality_distance / quality_total

if __name__ == '__main__':
    args = Configs().parse()
    if args.dump_xmodel:
        args.device='cpu'

    if args.device=='cpu':
        device = torch.device("cpu")
    else:
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = torch.device("cuda")
 
    train_loader, test_loader = data.get_data(args)
    net = model.get_model()
    net = net.to(device)
    if args.pretrained:
        new_dict = {}
        if os.path.isfile(args.pretrained):
            net_dict = net.state_dict()
            print("=> loading pretrained model '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location=device)
            #pretrained_dict = checkpoint['state_dict']
            for k, v in checkpoint.items(): 
                if k in net_dict.keys():
                    new_dict[k] = v
            net_dict.update(new_dict)
            net.load_state_dict(net_dict)
        else:
            print("=> no pretrained found at '{}'".format(args.pretrained))
    if args.quant_mode == 'float':
        if not args.evaluate:
            name = 'face_quality_{}'.format(time.strftime('%m%d_%H%M', time.localtime()))
            log_dir = 'log/'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            logname = 'log/log-{}.log'.format(name)
            Log.Log(logname, 'w+', 1)
            TRAIN_PARAMETER = '''\
            # TRAIN_PARAMETER
            ## loss
            l1loss
            ## optimizer
            SGD: base_lr %f momentum %f weight_decay %f
            ## lr_policy
            MultiStepLR: milestones [%s] gamma %f epochs %d
            size: %s
            '''%(
            args.base_lr,
            args.momentum,
            args.weight_decay,
            ', '.join(str(e) for e in args.milestones),
            args.gamma,
            args.epochs,
            args.size,
            )
            print(TRAIN_PARAMETER)
            train_net(args, net, train_loader, test_loader)
        else:
            error_p, error_q = eval_net(net, test_loader)
            print('quality_error: {:.4}'.format(error_q))
    else:
        # get quantizable model
        ################################################################
        input = torch.randn([1, 3, 80, 60])
        input = input.to(device)
        quantizer = torch_quantizer(args.quant_mode, net, (input), device=device)
        quant_net = quantizer.quant_model
        error_p, error_q = eval_net(quant_net, test_loader)
        print('quality_error: {:.4}'.format(error_q))

        # handle quantization result
        if args.quant_mode == 'calib':
            quantizer.export_quant_config()
        if args.quant_mode == 'test' and args.dump_xmodel:
            #deploy_check= True if args.dump_golden_data else False
            dump_xmodel("quantize_result", deploy_check=True)
