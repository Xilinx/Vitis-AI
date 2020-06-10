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

# This code is based on: https://github.com/nutonomy/second.pytorch.git
# 
# MIT License
# Copyright (c) 2018 
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
import random
import torch
import numpy as np
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

cudnn.benchmark = True
import torch.nn as nn
import logging
import argparse
from code.configs.model_config import Configs
def main(args):
    # network
    from code.models.enet_xilinx import ENet
    print('====> Bulid Networks...')
    net = ENet(args.num_classes)

    if os.path.isfile(args.weight):
        state_dict = torch.load(args.weight)  
        net.load_state_dict(state_dict['state_dict'])
        print("====> load weights sucessfully!!!")
    else:
        logging.error('can not find the checkpoint.')
        exit(-1)
    # get quantizable model
    ###########################################################
    x = torch.randn(1,3,512,1024)
    quantizer = torch_quantizer(args.quant_mode, net, (x))
    net = quantizer.quant_model.cuda()
    ##########################################################
    net = net.cuda()
    net.eval()

    if args.eval:
        print('====> Evaluation mIoU')
        eval_miou(args, net)    
    else:
        print('====> Prediction for demo')
        demo(args, net)
    # export calibration quant info (quant_mode = 1)
    quantizer.export_quant_config()
    # generate deploy model(quant_mode=2)
    dump_xmodel('quantize_result')

def eval_miou(args, net):
    from code.utils import evaluate
    from code.data_loader import cityscapes as citys
    from torch.utils.data import DataLoader
    import torchvision.transforms as standard_transforms
    # data
    print('====> Bulid Dataset...')

    assert args.dataset in ['cityscapes', 'camvid', 'coco', 'voc']
    val_set = citys.CityScapes(root=args.data_root, quality='fine', mode='val', size=(1024, 512))
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, num_workers=16, shuffle=False)

    inputs_all, gts_all, predictions_all = [], [], []
    with torch.no_grad():
        for vi, data in enumerate(val_loader):
            inputs, gts = data
            inputs = inputs.cuda()
            gts = gts.cuda()
            outputs = net(inputs)
            if outputs.size()[2:] != gts.size()[1:]:
                outputs = F.interpolate(outputs, size=gts.size()[1:], mode='bilinear', align_corners=True)           
            predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
            inputs_all.append(inputs.data.cpu())
            gts_all.append(gts.data.cpu().numpy())
            predictions_all.append(predictions)

        gts_all = np.concatenate(gts_all)
        predictions_all = np.concatenate(predictions_all)
        acc, acc_cls, ious, mean_iu, fwavacc = evaluate(predictions_all, gts_all, args.num_classes, args.ignore_label)
        print('mIoU(%): {}'.format(mean_iu * 100))


def demo(args, net):
    import cv2, glob
    from PIL import Image
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.savedir)
    # read all the images in the folder
    image_list = glob.glob(args.demo_dir + '*')

    # color pallete
    pallete = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32 ]

    mean = [.485, .456, .406]
    std =  [.229, .224, .225]

    for i, imgName in enumerate(image_list):
        name = imgName.split('/')[-1]
        img = cv2.imread(imgName).astype(np.float32)
        H, W = img.shape[0], img.shape[1]
        # image normalize
        img = cv2.resize(img, (args.input_size[0], args.input_size[1]))
        img =  img / 255.0
        for j in range(3):
            img[:, :, j] -= mean[j]
        for j in range(3):
            img[:, :, j] /= std[j]
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
        with torch.no_grad():
            img_variable = img_tensor.cuda()
            outputs = net(img_variable)
            if outputs.size()[-1] != W:
                outputs = F.interpolate(outputs, size=(H, W), mode='bilinear', align_corners=True)

            classMap_numpy = outputs[0].max(0)[1].byte().cpu().data.numpy()
            classMap_numpy = Image.fromarray(classMap_numpy)

            name = imgName.split('/')[-1]
            classMap_numpy_color = classMap_numpy.copy()
            classMap_numpy_color.putpalette(pallete)
            classMap_numpy_color.save(os.path.join(args.save_dir, 'color_' + name))

if __name__ == '__main__':
    args = Configs().parse()
    torch.manual_seed(args.seed)    
    main(args)

