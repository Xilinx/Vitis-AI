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

# PART OF THIS FILE AT ALL TIMES.

#!/usr/bin/python2
#-*-coding:utf-8-*-
import numpy as np
import cv2
import time
import argparse
import os
import commands
import pdb
import math

import sys
sys.path.insert(0,'../../../../../caffe-xilinx/python')
sys.path.insert(0,'../../../../../caffe-xilinx/python/caffe')

import caffe

def getOutput(args):
    fp = open(args.testImgList)
    lines = fp.readlines()
    fp.close()
    i = 0
    l2_loss = 0.0
    while i < len(lines):
        image_name = args.inputImgPath + lines[i].strip().split(' ')[0] + '.jpg'
        image = cv2.imread(image_name)
        write_image = image.copy()
        h, w, _ = image.shape
        # generate image data for input
        input_image = image.astype(np.float)
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = input_image - args.mean_value
        input_image = input_image * args.scale
        # load data
        net.blobs['data'].data[...] = input_image
        # forward and time count
        output = net.forward()
        #generate result
        # args
        points = output[args.points][0, ...]
        #sex = output[args.sex][0, ...]
        #age = output[args.age][0, ...]
        gts = lines[i].strip().split(' ')[1:]
        # transfer points
        for k in range(5):
            #pdb.set_trace()
            #print points[k]*96, points[k+5]*72
            points_x_loss = math.sqrt(pow(points[k] - float(gts[k])/args.im_width, 2))
            points_y_loss = math.sqrt(pow(points[k+5] - float(gts[k+5])/args.im_height, 2))
            l1_loss = l1_loss + 0.5*(points_x_loss + points_y_loss) 
        i = i + 1
    l1_loss = l1_loss/float(i)

    return l2_loss

if __name__ == "__main__":

    # load argparse

    parser = argparse.ArgumentParser(description = 'analysis landmark model, must be deploy, not train')
    parser.add_argument('--model', help = 'model structure for landmarks and attribute', default='../../float/test.prototxt')
    parser.add_argument('--weights', help = 'model weights landmark', default='../../float/trainval.caffemodel')
    parser.add_argument('--testImgList', type = str, help = 'test image list', default='/group/modelzoo/test_dataset/faceKeypointsAttr/align_test_crop_list.txt')
    parser.add_argument('--inputImgPath', type = str, help = 'input image path', default='/group/modelzoo/test_dataset/faceKeypointsAttr/align_test_crop/')
    parser.add_argument('--gpu', type = int, help = 'select gpu', default=0)
    args = parser.parse_args()
    
    args.points = 'fc6_points'
    #args.sex = 'fc6_sex'
    #args.age = 'fc6_age'
    args.im_height = 96
    args.im_width = 72
    args.mean_value = 127.5
    args.scale = 0.00784315
    loss_weight = 168

    print('model define file: %s' % args.model)
    print('model weights file: %s' % args.weights)
    print('gpu id: %d' % args.gpu)
    print('model points output blob name: %s' % args.points)
    #print 'model sex output blob name: %s' % args.sex
    #print 'model age output blob name: %s' % args.age

    # load caffe
    os.environ['GLOG_minloglevel'] = '3'

    # load network
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    model_def = args.model
    model_weights = args.weights
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    l1_loss = 168 * getOutput(args)
    print('weighted points-l1_loss is %.4f'%(l1_loss))

