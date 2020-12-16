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
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils import data
import numpy as np

sys.path.insert(0,'..')
from dataset.datasets import ChaosCTDataSet as DataSet

from utils.train_options import TrainOptions
from networks.build_model import NetModel

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def main():
    args = TrainOptions().initialize()
    model = NetModel(args)
    trainloader, valloader, save_steps, H, W = get_dataset(args, 'ChaosCT')
    batch =  iter(trainloader)
    
    max_loss = 10.0
    max_dice = 0
    for iteration in range(0, args.num_steps):
        model.adjust_learning_rate(args.lr, model.solver, iteration)
        model.solver.zero_grad()
        loss = 0.0
        inputs, labels = set_input(next(batch))
        pred = model.parallel_student.train()(inputs)
        loss = model.criterion(pred, labels)
        model.loss = loss.item()
        loss.backward()
        model.solver.step()    

        if iteration % 20 == 0:
            logging.info('iteration:{:5d} lr:{:.6f} loss:{:.5f}'.format( \
                        iteration, model.solver.param_groups[-1]['lr'], model.loss))

        if (iteration > 1) and ((iteration % save_steps == 0) and (iteration > args.num_steps - 2000)) or (iteration == args.num_steps - 1):
            mean_IU, IU_array, Dice = model.evalute_model(model.student, valloader, args.gpu, H, W, args.classes_num, args.ignore_index, True)
            logging.info('[val with {}x{}] mean_IU:{:.6f} Dice:{:.6f}'.format(H, W, mean_IU, Dice))
            if Dice > max_dice:
                max_dice = Dice
                model.save_ckpt(iteration, Dice, 'ChaosCT')
                logging.info('save ckpt at {} with Dice:{:.6f}'.format(iteration, max_dice))

def set_input(data):
    images, labels, _, _ = data
    images = images.cuda()
    labels = labels.long().cuda()
    if torch.version == "0.3":
        images = Variable(images)
        labels = Variable(labels)
    return images, labels

def get_dataset(args, data_set):
    h, w = map(int, args.input_size.split(','))
    trainloader= data.DataLoader(DataSet(root=args.data_dir, list_path='./dataset/lists/train_img_seg.txt', \
                      max_iters=args.num_steps*args.batch_size, crop_size=(h, w), mean=IMG_MEAN, \
                      scale=args.random_scale, mirror=args.random_mirror, ignore_label=args.ignore_index), \
                      batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    valloader = data.DataLoader(DataSet(root=args.data_dir, list_path='./dataset/lists/val_img_seg.txt', \
                      crop_size=(512, 512), mean=IMG_MEAN, scale=True, mirror=False, ignore_label=args.ignore_index), \
                      batch_size=1, shuffle=False, pin_memory=True)
    save_steps = int(479/args.batch_size)
    H, W = 512, 512
    
    return trainloader, valloader, save_steps, H, W

if __name__ == '__main__':
    main()

