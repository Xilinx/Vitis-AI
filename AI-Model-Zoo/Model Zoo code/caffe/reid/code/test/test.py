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
from utils.eval_func import eval_func
from utils.common import normalize, cosine_distance, bn, get_batch_images
from market1501 import Market1501

caffe.set_mode_gpu()
caffe.set_device(0)

class ReidModel(object):

    bn_mean = np.load('../../data/bn_params/bn_mean.npy')
    bn_var = np.load('../../data/bn_params/bn_var.npy')
    bn_weight = np.load('../../data/bn_params/bn_weight.npy')
    
    def __init__(self, prototxt_path, caffemodel_path):
        self.net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

    def forward(self, img_paths):
        ims = get_batch_images(img_paths)
        self.net.blobs['data'].reshape(len(ims), 3, 160, 80)    
        self.net.blobs['data'].data[...] = ims
        out = self.net.forward()
        feature = out['View_1']
        feature = bn(feature, ReidModel.bn_mean, ReidModel.bn_var, ReidModel.bn_weight)
        feature = [normalize(a) for a in feature]
        return feature

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for evaluting reid model on marker1501')
    parser.add_argument('--prototxt_path', help='prototxt path')
    parser.add_argument('--caffemodel_path', help='caffemodel path')
    parser.add_argument('--dataset_root', help='root to market1501 folder')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    args = parser.parse_args()

    # prepare market1501 dataset
    market1501 = Market1501(root=args.dataset_root)
    query = market1501.query
    gallery = market1501.gallery
    test_set = query + gallery
    
    model = ReidModel(args.prototxt_path, args.caffemodel_path)

    feat_list = []
    pids = []
    camids = []
    count =0
    while count < len(test_set)-1:
        cur_batch_size = min(args.batch_size, len(test_set)-count)
        img_paths, person_ids, cam_ids = zip(*test_set[count:count+cur_batch_size])
        pids.extend(person_ids)
        camids.extend(cam_ids)

        # network forward
        feat = model.forward(img_paths)
        feat_list = feat_list + feat

        count += cur_batch_size

    query_feats = feat_list[:len(query)]
    q_pids = pids[:len(query)]
    q_camids = camids[:len(query)]
    gallery_feats = feat_list[len(query):]
    g_pids = pids[len(query):]
    g_camids = camids[len(query):]
    
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    dismat = np.zeros((len(query_feats), len(gallery_feats)))
    for i in range(len(query_feats)):
        for j in range(len(gallery_feats)):
            dismat[i,j] = cosine_distance(query_feats[i], gallery_feats[j])
    cmc, mAP = eval_func(dismat,  q_pids, g_pids, q_camids, g_camids)
    print('[Reid evaluation]')
    print(args)
    print("Evaluation results -------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in [1, 5, 10]:
         print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1])) 
    print("--------------------------")
