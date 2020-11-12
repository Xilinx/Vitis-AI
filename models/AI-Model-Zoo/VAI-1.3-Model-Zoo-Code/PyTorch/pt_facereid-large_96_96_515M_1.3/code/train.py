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


# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import sys
import torch
from torch import nn

import network
from core.config import opt, update_config
from core.loader import get_data_provider
from core.solver import Solver
from utils.loss import TripletLoss
from utils.lr_scheduler import LRScheduler
from ipdb import set_trace

FORMAT = '[%(levelname)s]: %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    stream=sys.stdout
)


def train(args):
    logging.info('======= user config ======')
    logging.info(print(opt))
    logging.info(print(args))
    logging.info('======= end ======')

    train_data, test_data, num_query, num_class = get_data_provider(opt, args.dataset, args.dataset_root)
    logging.info('Setting the network, num_Class:{}'.format(num_class))
    
    net = getattr(network, args.network)(num_class, opt.network.last_stride)
    if args.resume:
        checkpoint = torch.load(args.resume)
        model_dict = net.state_dict()
        pretrain_dict = checkpoint['state_dict']
        for i in pretrain_dict:
            net.state_dict()[i].copy_(pretrain_dict[i])
    if opt.device.type == 'cuda':
        net = nn.DataParallel(net)
        net = net.to(opt.device)
    net.eval()

    optimizer = getattr(torch.optim, opt.train.optimizer)(net.parameters(), lr=opt.train.lr, weight_decay=opt.train.wd)
    ce_loss = nn.CrossEntropyLoss()
    triplet_loss = TripletLoss(margin=opt.train.margin)

    def ce_loss_func(scores, feat, labels):
        ce = ce_loss(scores, labels)
        return ce

    def tri_loss_func(scores, feat, labels):
        tri = triplet_loss(feat, labels)[0]
        return tri

    def ce_tri_loss_func(scores, feat, labels):
        ce = ce_loss(scores, labels)
        triplet = triplet_loss(feat, labels)[0]
        return ce + triplet


    if opt.train.loss_fn == 'softmax':
        loss_fn = ce_loss_func
    elif opt.train.loss_fn == 'triplet':
        loss_fn = tri_loss_func
    elif opt.train.loss_fn == 'softmax_triplet':
        loss_fn = ce_tri_loss_func
    else:
        raise ValueError('Unknown loss func {}'.format(opt.train.loss_fn))

    lr_scheduler = LRScheduler(base_lr=opt.train.lr, step=opt.train.step,
                               factor=opt.train.factor, warmup_epoch=opt.train.warmup_epoch,
                               warmup_begin_lr=opt.train.warmup_begin_lr)

    mod = Solver(opt, net)
    mod.fit(train_data=train_data, test_data=test_data, num_query=num_query, optimizer=optimizer,
            criterion=loss_fn, lr_scheduler=lr_scheduler)


def main():
    parser = argparse.ArgumentParser(description='reid model training')
    parser.add_argument('--dataset', type=str, default = 'market1501', 
                       help = 'set the dataset for training')
    parser.add_argument('--dataset_root', type=str, default = '../data/face_reid', 
                       help = 'dataset path')
    parser.add_argument('--config_file', type=str, default=None, required=True,
                        help='Optional config file for params')
    parser.add_argument('--save_dir', type=str, default=None, required=True,
                        help='model save checkpoint directory')
    parser.add_argument('--resume', type = str, default = None,
                        help = 'load model to continue training')
    parser.add_argument('--device', type=str, default='gpu', choices=['gpu','cpu'],
                        help='set running device')
    parser.add_argument('--gpu', type = str, default = '0,1')
    parser.add_argument('--network', type=str, default='Baseline')
    parser.add_argument('--VAR_weight', type=float, default=0.01)

    args = parser.parse_args()
    if args.config_file is not None:
        update_config(args.config_file)
    opt.misc.save_dir = args.save_dir
    if args.device=='gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
        logging.info('Your training device is set to [CPU]')

    train(args)


if __name__ == '__main__':
    main()
