import argparse
import pdb
import os
import shutil
import time
import math
import copy
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from pytorch_nndct.apis import torch_quantizer,dump_xmodel
import numpy as np

import sys
sys.path.append('../models')
sys.path.append('../utils')

from load_imglist import ImageList
import face_model
import lfw
import evaluate
import image_transforms


parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_am',
                    help='model architecture: (default: vgg19)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--testset_images', default='../../data/test/lfw/images', type=str, metavar='PATH',
                    help='path to testset (default: none)')
parser.add_argument('--testset_pairs', default='../../data/test/lfw/pairs.txt', type=str, metavar='PATH',
                    help='path to testset (default: none)')
parser.add_argument('--testset_list', default='../../data/test/lfw/lfw.txt', type=str, metavar='PATH',
                    help='path to testset (default: none)')
parser.add_argument('--fea-dim', default=512, type=int)
parser.add_argument('--num_classes', default=180855, type=int)
parser.add_argument('--scale', default=80, type=float, help='scale for amsoftmax')
parser.add_argument('--margin', default=0.4, type=float, help='margin for amsoftmax or lamsoftmax')
parser.add_argument('--flip', default=False, action='store_true', help='use flip operation')
parser.add_argument('--gpu', default=True, action='store_true', help='use gpu')
parser.add_argument('--gpu_id', type=str, default='0,1')
parser.add_argument('--quant_mode', type=int, default='1')
args = parser.parse_args()

def main():
    
    model = face_model.__dict__[args.arch.replace('_am', '')](args.num_classes, wn=False, fn=False, fea_dim=args.fea_dim)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        model = model.to(torch.device('cuda:{}'.format(args.gpu_id)))
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        # load part of state_dict
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        new_dict = {}
        for k, v in pretrained_dict.items():
            k1 = k.split('module.')[1]
            if not k1 in model_dict:
                continue
            else:
                new_dict[k1] = v
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    else:
        print("wrong checkpoint path", checkpoint)
    # get quantizable model
    ######################################################
    x = torch.randn([args.batch_size, 3, 112, 96]).cuda()
    quantizer = torch_quantizer(args.quant_mode, model, (x))
    model = quantizer.quant_model.cuda()
    ######################################################

    accuracy = evaluate.eval_lfw(model, args.testset_images, args.testset_list, args.testset_pairs, batch_size=args.batch_size, pth=False, fea_dim=args.fea_dim, gpu=args.gpu)
    print("accuracy is %.4f"%(accuracy))
    
    # export calibration quant info (quant_mode=1)
    quantizer.export_quant_config()
    # generate deploy model(quant_mode=2)
    dump_xmodel('quantize_result')


if __name__ == '__main__':
    main()
