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

import os
import sys

import caffe
import numpy as np
from utils.eval_func import eval_func
from utils.common import normalize, cosine_distance, bn, get_batch_images
from utils.data_read import Market1501
import argparse


class ReidModel(object):

    bn_mean = np.load('./data/bn_params/bn_mean.npy')
    bn_var = np.load('./data/bn_params/bn_var.npy')
    bn_weight = np.load('./data/bn_params/bn_weight.npy')
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, help='Path to market1501 directory', default="./data")
    args = vars(parser.parse_args())
    
    dataset_root = args["img_dir"] # root of market1501 foloder
    prototxt_path = 'xfdnn_auto_cut_deploy.prototxt'
    caffemodel_path = 'quantize_results/deploy.caffemodel'
    batch_size = 96

    # prepare market1501 dataset
    market1501 = Market1501(root=dataset_root)
    query = market1501.query
    gallery = market1501.gallery
    test_set = query + gallery
    
    model = ReidModel(prototxt_path, caffemodel_path)

    feat_list = []
    pids = []
    camids = []
    count = 0
    print("[INFO] Working on {} images..".format(len(test_set)))
    while count < len(test_set)-1:
        cur_batch_size = min(batch_size, len(test_set)-count)
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
    print('Caffemodel: {}'.format(caffemodel_path)) 
    print('Prototxt: {}'.format(prototxt_path)) 
    print("Evaluation results -------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in [1, 5, 10]:
         print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1])) 
    print("--------------------------")
