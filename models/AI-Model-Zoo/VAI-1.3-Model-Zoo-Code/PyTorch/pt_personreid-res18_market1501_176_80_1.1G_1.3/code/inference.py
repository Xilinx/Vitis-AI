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

from network.baseline import Baseline
from core.config import opt, update_config
import torch.nn.functional as F
from ipdb import set_trace


def load_reid_model(model_path):
    '''
    function to load my trained reid model
    '''
    net = Baseline(opt.network.backbone, last_stride=opt.network.last_stride)

    checkpoint = torch.load(model_path)
    model_dict = net.state_dict()
    for i in checkpoint:
        if 'classifier' in i or 'fc' in i:  continue
        net.state_dict()[i].copy_(checkpoint[i])
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
    parser = argparse.ArgumentParser(description='Person reid demo')
    parser.add_argument('--data_dir', default = '../data/test_imgs/', 
                        help='directory to test data')
    parser.add_argument('--model_path', required=True, 
                        help='path to model')
    parser.add_argument('--config_file', type=str,  
                        help='Optional config file for params')
    parser.add_argument('--gpu', default = '1')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    update_config(args.config_file)
    img_dir = args.data_dir
    query = ['1.jpg']
    gallery = ['2.jpg',
                '3.jpg',
                '4.jpg',
                '5.jpg',
                '6.jpg',
                '7.jpg']
    img_list = query + gallery
    resize_wh = opt.aug.resize_size #[128,256]
        
    imgs = [io.imread(os.path.join(img_dir, imgname)) for imgname in img_list]
    imgs = np.asarray([im_preprocess(cv2.resize(p, (resize_wh[1], resize_wh[0]))) for p in imgs], dtype=np.float32)
    reid_model = load_reid_model(args.model_path)
    print('[INFO] Load model path: {}'.format(args.model_path))

    with torch.no_grad():
        im_var = Variable(torch.from_numpy(imgs))
        im_var = im_var.cuda()
        feats = F.normalize(reid_model(im_var)).cpu()
    
    q_feats = feats[:len(query)]
    g_feats = feats[len(query):]
    distmat = cdist(q_feats, g_feats)
    print('[INFO] Query-gallery distance:')
    for i,q in enumerate(query):
        for j,g in enumerate(gallery):
            print('  {} - {} distance: {:.2f}'.format(q,g,distmat[i][j]))
        
    for i in range(len(query)):
        g_idx = np.where(distmat[i]==np.min(distmat[i]))[0][0]
        print('[INFO] Gallery image {} is the same id as query image {}\n'.format(gallery[g_idx], query[i]))
