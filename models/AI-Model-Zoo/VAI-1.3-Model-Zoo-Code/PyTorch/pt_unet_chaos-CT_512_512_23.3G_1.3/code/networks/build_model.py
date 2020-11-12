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

import argparse
import logging
import os,sys
import pdb
from torch.autograd import Variable
import os.path as osp
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import resource
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0,'..')
from networks.UNet import unet
from utils.criterion import Criterion
from networks.evaluate import evaluate_main
from utils.flops_compute import add_flops_counting_methods
from utils.utils import *
torch_ver = torch.__version__[:3]

def compute_flops(model, input=None):
    input = input if input is not None else torch.Tensor(1, 3, 224, 224)
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()

    _ = model(input)

    flops = model.compute_average_flops_cost()  
    flops = flops/ 1e9 
    return np.float(flops)


def warmup_lr(base_lr, it, warmup_iters=500, warmup_factor=float(1.0/3.0), method='linear'):
    if method == 'constant':
        factor = warmup_factor
    elif method == 'linear':
        alpha = float(it / warmup_iters)
        factor = warmup_factor * (1 -  alpha) + alpha
    else:
        print('error warmup method')
    lr = base_lr * factor
    return lr

def get_model(args):
    h, w = map(int, args.input_size.split(','))
    model = unet(n_classes=2)
    logging.info(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, h, w))
    logging.info('FLops = {}G with H*W = {} x {}'.format(flops, h, w))
    return model

class NetModel():
    def name(self):
        return '2D-UNet(light-weight) on Chaso-CT Dataset'

    def DataParallelModelProcess(self, model, is_eval = 'train', device = 'cuda'):
        parallel_model = torch.nn.DataParallel(model)#DataParallelModel(model)
        if is_eval == 'eval':
            parallel_model.eval()
        elif is_eval == 'train':
            parallel_model.train()
        else:
            raise ValueError('is_eval should be eval or train')
        parallel_model.float()
        parallel_model.to(device)
        return parallel_model

    def DataParallelCriterionProcess(self, criterion, device = 'cuda'):
        criterion.cuda()
        return criterion

    def __init__(self, args):
        #cudnn.enabled = True
        self.args = args
        device = args.device
        student = get_model(args)
        load_S_model(args, student, False)
        print_model_parm_nums(student, 'student_model')
        self.parallel_student = self.DataParallelModelProcess(student, 'train', device)
        self.student = student

        self.solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, self.student.parameters()), 'initial_lr': args.lr}], \
                                                          args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        class_weights = [1, 12] 
        self.criterion = self.DataParallelCriterionProcess(Criterion(weight=class_weights, ignore_index=255))
        self.loss = 0.0

        if not os.path.exists(args.ckpt_path):
            os.makedirs(args.ckpt_path)

    def AverageMeter_init(self):
        self.parallel_top1_train = AverageMeter()
        self.top1_train = AverageMeter()

    def lr_poly(self, base_lr, iter, max_iter, power):
        
        return base_lr*((1-float(iter)/max_iter)**(power))
            
    def adjust_learning_rate(self, base_lr, optimizer, i_iter):
        args = self.args
        if args.warmup:
            if i_iter < 500:
                lr = warmup_lr(base_lr, i_iter)
            else:
                lr = base_lr*((1-float(i_iter-500)/(args.num_steps-500))**(args.power))
        else:
            lr = self.lr_poly(base_lr, i_iter, args.num_steps, args.power)
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def evalute_model(self, model, loader, gpu_id, h, w, num_classes, ignore_label, whole):
        mean_IU, IU_array, dice = evaluate_main(model=model,loader = loader, gpu_id = gpu_id, h=h, w=w, num_classes = num_classes,\
                                                ignore_label = ignore_label,whole = whole)
        return mean_IU, IU_array, dice 

    def print_info(self, epoch, step):
        logging.info('step:{:5d} lr:{:.6f} loss:{:.5f}'.format(
                        step, self.solver.param_groups[-1]['lr'], self.loss))

    def __del__(self):
        pass

    def save_ckpt(self, step, acc, name):
        torch.save(self.student.state_dict(),osp.join(self.args.ckpt_path, name+'_'+str(step)+'_'+str(acc)+'.pth'))  




