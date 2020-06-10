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

from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2

import sys
sys.path.append('../models')
sys.path.append('../utils')
from load_imglist import ImageList
import lfw
import image_transforms

def get_features(model, imgroot, imglist, batch_size, pth=False, nonlinear='relu', flip=True, fea_dim=512, gpu=True):
    model.eval()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    gammabalance = image_transforms.GammaBalance()
    transform_ops = [ 
            transforms.ToTensor(),
            normalize,
    ]
    transform=transforms.Compose(transform_ops)

    val_loader = torch.utils.data.DataLoader(
        ImageList(root=imgroot, fileList=imglist, 
            transform=transform, flip=flip),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    with open(imglist) as f:
        imgnames = f.readlines()
    fea_dict = dict()
    out_dict = dict()
    fea_list = list()
    start = time.time()
    norms = []
    for i, (input, target) in enumerate(val_loader):
        if gpu:
            input      = input.cuda()
            target     = target.cuda()
        with torch.no_grad():
            input_var  = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
        
        l = input.size()[0]
        if flip:
            assert(l % 2 == 0)
        features = model(input_var)
        features = features.data.cpu()
        # print(features.shape)
        end = time.time() - start
        # if i == len(val_loader) - 1:
        if i % 10000 == 0:
            print("({}/{}). Time: {}".format((i+1)*batch_size, len(val_loader)*batch_size, end))
        le = l // 2 if flip else l
        for j in range(le):
            idx = i*batch_size//2+j if flip else i*batch_size+j
            imgname = imgnames[idx].strip().split()[0]
            name = imgname.split('/')[-1]
            jdx = j*2 if flip else j
            f1 = features.numpy()[jdx]
            if flip:
                f2 = features.numpy()[jdx+1]
                #feature = f1 + f2
                feature = np.append(f1, f2)
                assert(feature.shape[0] == f1.shape[0] + f2.shape[0])
            else:
                feature = f1.copy()
            norm = np.linalg.norm(feature)
            fea_dict[name] = feature / norm
            fea_list.append(feature / norm)
            norms.append(norm)
    norms  =np.array(norms)
    print('xnorm: {}\tmaxnorm: {}\tminnorm: {}\tstd: {}'.format(np.mean(norms), np.max(norms), np.min(norms), np.std(norms)))
    return fea_dict, fea_list

def eval_lfw(model, imgroot, imglist, pair_path, batch_size=64, pth=False, nonlinear='relu', flip=True, fea_dim=512, gpu=True):
    fea_dict, fea_list = get_features(model, imgroot, imglist, batch_size, pth, nonlinear, flip, fea_dim, gpu)
    acc = lfw.test(fea_dict, pair_path, write=False)
    return acc

