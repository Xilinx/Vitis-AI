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


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from skimage import io
import cv2
from scipy.spatial.distance import cdist 
import argparse

import torch.nn.functional as F 
from network.resnet18 import Resnet18
from network.resnet_small import Resnetsmall
from ipdb import set_trace


def load_reid_model(net_name, model_path):
    '''
    function to load my trained reid model
    '''
    assert net_name == 'facereid_small' or net_name == 'facereid_large'
    if net_name == 'facereid_small': 
        net = Resnetsmall()
    else:
        net = Resnet18()

    checkpoint = torch.load(model_path)
    pretrain_dict = checkpoint['state_dict']
    model_dict = net.state_dict()
    for i in pretrain_dict:
        if 'classifier' in i or 'fc' in i:  continue
        net.state_dict()[i].copy_(pretrain_dict[i])
    net = net.cuda()
    net.eval()
    return net 

def im_preprocess(image):
    image = np.asarray(image, np.float32)
    image /=255.0
    image -= np.array([0.485, 0.456, 0.406]).reshape(1, 1, -1)
    image /= np.array([0.229, 0.224, 0.225]).reshape(1, 1, -1)
    image = np.transpose(image, [2,0,1]) # data format: hwc->chw
    return image

def parse_args():
    parser = argparse.ArgumentParser(description='FaceReid demo')
    parser.add_argument('--data_dir', default = '../data/test_imgs/', help='directory to test data')
    parser.add_argument('--network', choices=['facereid_small', 'facereid_large'],
                        help='set network, choose from [facereid_small, facereid_large] ')
    parser.add_argument('--model_path', required=True, help='path to model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    img_dir = args.data_dir
    query = ['1.jpg']
    gallery = ['2.jpg',
                '3.jpg',
                '4.jpg',
                '5.jpg',
                '6.jpg',
                '7.jpg']
    img_list = query + gallery

    resize_wh = 80 if args.network == 'facereid_small' else 96
        
    imgs = [io.imread(os.path.join(img_dir, imgname)) for imgname in img_list]
    imgs = np.asarray([im_preprocess(cv2.resize(p, (resize_wh, resize_wh))) for p in imgs], dtype=np.float32)
    reid_model = load_reid_model(args.network, args.model_path)
    print('[INFO] Load model: {}, model path: {}'.format(args.network, args.model_path))

    with torch.no_grad():
        im_var = Variable(torch.from_numpy(imgs))
        im_var = im_var.cuda()
        feats = F.normalize(reid_model(im_var)).cpu()
    
    q_feats = feats[:len(query)]
    g_feats = feats[len(query):]
    distmat = cdist(q_feats, g_feats)
    print('[INFO] Query-gallery-distance matrix: \n  ', distmat)
        
    for i in range(len(query)):
        g_idx = np.where(distmat[i]==np.min(distmat[i]))[0][0]
        print('[INFO] Link query image {} with gallery image {} as same id'.format(query[i], gallery[g_idx]))
