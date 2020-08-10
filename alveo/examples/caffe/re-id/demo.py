#!/usr/bin/env python
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
import caffe
import numpy as np
from utils.common import normalize, cosine_distance, bn, process_image
import argparse

np.set_printoptions(threshold=sys.maxsize)

# Need to create derived class to clean up properly
class Net(caffe.Net):
  def __del__(self):
    for layer in self.layer_dict:
      if hasattr(self.layer_dict[layer],"fpgaRT"):
        del self.layer_dict[layer].fpgaRT

class ReidModel(object):
    bn_mean = np.load('./data/bn_params/bn_mean.npy')
    bn_var = np.load('./data/bn_params/bn_var.npy')
    bn_weight = np.load('./data/bn_params/bn_weight.npy')
    
    def __init__(self, prototxt_path, caffemodel_path):
        self.net = Net(prototxt_path, caffemodel_path, caffe.TEST)

    def forward(self, img_path):
        im = process_image(img_path)
        self.net.blobs['data'].data[...] = im
        out = self.net.forward()
        feature = out['View_1'][0]
        feature = bn(feature, ReidModel.bn_mean, ReidModel.bn_var, ReidModel.bn_weight)
        feature = normalize(feature)
        return feature
    
    def __del__(self):
        self.net.__del__()

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
    del model
    print('[Reid demo]')
    print('Caffemodel: {}'.format(caffemodel_path)) 
    print('Prototxt: {}'.format(prototxt_path)) 
    print('Distance matrix:') 
    print(dismat)
    print('Over -------------------')
