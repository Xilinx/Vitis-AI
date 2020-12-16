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
sys.path.insert(0, '../')

import argparse
import caffe
import numpy as np
from utils.common import normalize, cosine_distance, bn, process_image

#caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(0)

class ReidModel(object):
    bn_mean = np.load('../../data/bn_params/bn_mean.npy')
    bn_var = np.load('../../data/bn_params/bn_var.npy')
    bn_weight = np.load('../../data/bn_params/bn_weight.npy')
    
    def __init__(self, prototxt_path, caffemodel_path):
        self.net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

    def forward(self, img_path):
        im = process_image(img_path)
        self.net.blobs['data'].data[...] = im
        out = self.net.forward()
        feature = out['View_1'][0]
        feature = bn(feature, ReidModel.bn_mean, ReidModel.bn_var, ReidModel.bn_weight)
        feature = normalize(feature)
        return feature
           

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Script for getting a person reid demo')
    parser.add_argument('--prototxt_path', default='../../float/test.prototxt', help='prototxt path')
    parser.add_argument('--caffemodel_path', default='../../float/trainval.caffemodel', help='caffemodel path')
    args = parser.parse_args()

    # set paths here
    image_root = '../../data/test_imgs/'
    image_list = ['1.jpg',
                  '2.jpg',
                  '3.jpg',
                  '4.jpg',
                  '5.jpg',
                  '6.jpg',
                  '7.jpg']

    model = ReidModel(args.prototxt_path, args.caffemodel_path)
    feats= []
    for image_path in image_list:
        feat = model.forward(os.path.join(image_root, image_path))
        feats.append(feat)
    
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
