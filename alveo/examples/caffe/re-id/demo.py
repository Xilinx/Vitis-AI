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
from utils.common import normalize, cosine_distance, bn, process_image
import argparse

np.set_printoptions(threshold=sys.maxsize)

caffe.set_mode_cpu()

class ReidModel(object):
    bn_mean = np.load('./data/bn_params/bn_mean.npy')
    bn_var = np.load('./data/bn_params/bn_var.npy')
    bn_weight = np.load('./data/bn_params/bn_weight.npy')
    
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
    # set paths here
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_image', type=str, help="query image path")
    parser.add_argument('--test_image', type=str, help="test image path")
    args = vars(parser.parse_args())
    
    image_list = [args["query_image"],args["test_image"]]
    prototxt_path = 'xfdnn_auto_cut_deploy.prototxt'
    caffemodel_path = 'quantize_results/deploy.caffemodel'

    model = ReidModel(prototxt_path, caffemodel_path)
    feats= []
    for image_path in image_list:
        feat = model.forward(image_path)
        feats.append(feat)
    
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    dismat = np.zeros((len(feats), len(feats)))
    for i in range(len(feats)):
        for j in range(len(feats)):
            dismat[i,j] = cosine_distance(feats[i], feats[j])
    print('[Reid demo]')
    print('Caffemodel: {}'.format(caffemodel_path)) 
    print('Prototxt: {}'.format(prototxt_path)) 
    print('Distance matrix:') 
    print(dismat)
    print('Over -------------------')
