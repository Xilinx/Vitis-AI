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


# MIT License

# Copyright (c) 2019 Hengshuang Zhao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import sys
import copy
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import code.utils as utils
from code.utils.misc import save_checkpoint
from code.utils.metrics import batch_pix_accuracy, pixel_accuracy, batch_intersection_union
from code.utils.lr_scheduler import LR_Scheduler
from code.utils.metrics import *
from code.utils.parallel import DataParallelModel, DataParallelCriterion
from code.datasets import get_segmentation_dataset
from code.models import fpn  
from code.configs.model_config import Options
import logging
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable
#torch.backends.cudnn.benchmark = True

class Trainer():
    def __init__(self, args):
        self.args = args
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        trainset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train', root=args.data_folder, **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, split='val', mode ='val', root=args.data_folder, **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} 
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, \
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size, \
                                         drop_last=False, shuffle=False, **kwargs)
        self.nclass = args.num_classes
        self.best_pred = 0.0 
        # model
        model = fpn.get_fpn(nclass=args.num_classes, backbone=args.backbone, pretrained=False)
        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
        # optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(params_list, lr=args.lr, weight_decay=args.weight_decay)
        # criterions
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        # resuming checkpoint
        if args.weight is not None:
            if not os.path.isfile(args.weight):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cuda:0')
            checkpoint['state_dict'] = OrderedDict([(k[5:], v) if 'base' in k else (k, v) for k, v in checkpoint['state_dict'].items()])
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})" \
                  .format(args.weight, checkpoint['epoch']))
        # clear start epoch if fine-tuning
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, \
                                            args.epochs, len(self.trainloader), warmup_epochs=5)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader)
        for i, (image, target) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            image = image.cuda()
            target =  target.cuda()
            outputs = self.model(image)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            if isinstance(outputs, tuple):# for aux
                outputs = outputs[0]
            target = target.cuda()
            correct, labeled = batch_pix_accuracy(outputs.data, target)
            inter, union = batch_intersection_union(outputs.data, target, self.nclass)
            return correct, labeled, inter, union

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True).cuda()
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            else:
                with torch.no_grad():
                    image = image.cuda()
                    correct, labeled, inter, union = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, self.args, is_best)


if __name__ == "__main__":
    args = Options().parse()
    for key, val in args._get_kwargs():
        logging.info(key+' : '+str(val))
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
