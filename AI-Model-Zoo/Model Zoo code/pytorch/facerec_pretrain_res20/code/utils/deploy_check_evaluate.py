# (c) Copyright 2019 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.

from __future__ import print_function
import argparse
import os
import shutil
import time
import sys

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
from pytorch_nndct.apis import dump_xmodel

def get_features(model, imgroot, imglist, batch_size, pth=False, nonlinear='relu', flip=True, fea_dim=512, gpu=False):
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
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if gpu:
                input      = input.cuda()
                target     = target.cuda()
       # with torch.no_grad():
            input_var  = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
        
            l = input.size()[0]
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
            if i == 0:
                dump_xmodel("quantize_result", deploy_check=True)
                sys.exit()
    norms  =np.array(norms)
    print('xnorm: {}\tmaxnorm: {}\tminnorm: {}\tstd: {}'.format(np.mean(norms), np.max(norms), np.min(norms), np.std(norms)))
    return fea_dict, fea_list

def eval_lfw(model, imgroot, imglist, pair_path, batch_size=64, pth=False, nonlinear='relu', flip=True, fea_dim=512, gpu=True):
    fea_dict, fea_list = get_features(model, imgroot, imglist, batch_size, pth, nonlinear, flip, fea_dim, gpu)
    acc = lfw.test(fea_dict, pair_path, write=False)
    return acc

