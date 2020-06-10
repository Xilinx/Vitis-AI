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

import sys
sys.path.insert(0,'../../../../../caffe-xilinx/python')
sys.path.insert(0,'../../../../../caffe-xilinx/python/caffe')

import caffe

def getOutput(args):
    fp = open(args.testImgList)
    lines = fp.readlines()
    fp.close()
    i = 0
    while i < len(lines):
        image_name = lines[i].strip()
        image = cv2.imread(image_name)
        write_image = image.copy()
        h, w, _ = image.shape
        rect_num = int(lines[i + 1].strip())
        for j in range(rect_num):
            rect = lines[i + 2 + j].strip().split()
            rect = [float(v) for v in rect]
            # every rect mean a face, should be expanded
            xmin = max(0, int(rect[0] - 0.00 * rect[2]))
            xmax = min(w, int(rect[0] + 1.00 * rect[2]))
            ymin = max(0, int(rect[1] - 0.00 * rect[3]))
            ymax = min(h, int(rect[1] + 1.00 * rect[3]))
            crop_img = image[ymin:ymax, xmin:xmax].copy()
            resize_img = cv2.resize(crop_img, (args.im_width, args.im_height))
            # generate image data for input
            input_image = resize_img.astype(np.float)
            input_image = np.transpose(input_image, (2, 0, 1))
            input_image = input_image - args.mean_value
            input_image = input_image * args.scale
            # load data
            net.blobs['data'].data[...] = input_image
            # forward and time count
            time_start = time.time()
            output = net.forward()
            # generate result
            # args
            points = output[args.points][0, ...]
            #sex = output[args.sex][0, ...]
            #age = output[args.age][0, ...]
            t = time.time() - time_start
            # pre
            #sex = np.exp(sex)
            #sex = sex / np.sum(sex)
            #sex_flag = int(sex[1] > sex[0])
            #age = age * 60
            # log information in image
            # transfer points
            for k in range(5):
                points_x = int(xmin + (points[k] * (xmax - xmin)))
                points_y = int(ymin + (points[k + 5] * (ymax - ymin)))
                cv2.circle(write_image, (points_x, points_y), 2, (255, 0, 0), 2)
            # transfer sex and age
            font = cv2.FONT_HERSHEY_PLAIN
            text_x = int(rect[0])
            text_y = int(rect[1])
            text_xm = int(rect[0] + rect[2])
            text_ym = int(rect[1] + rect[3])
            #cv2.putText(write_image, 'score: %.3f' % rect[4], (text_x, text_y - 26), font, 1, (0, 0, 255), 1)
            #cv2.putText(write_image, '%s %0.2f' % (['female','male'][sex_flag], sex[sex_flag]), (text_x, text_y - 14), font, 1, (0, 0, 255), 1)
            #cv2.putText(write_image, 'age: %.2f' % age, (text_x, text_y - 2), font, 1, (0, 0, 255), 1)
            # rectangle for face
            #cv2.rectangle(write_image, (text_x, text_y), (text_xm, text_ym), (255, 0, 0), 2)
        image_name = image_name.split('/')[-1]
        cv2.imwrite(args.outputPath+'/output_%s' % image_name, write_image)
        i = i + 2 + rect_num

if __name__ == "__main__":

    # load argparse

    parser = argparse.ArgumentParser(description = 'analysis landmark model, must be deploy, not train')
    parser.add_argument('--model', help = 'model structure for landmarks and attribute', default='../../float/test.prototxt')
    parser.add_argument('--weights', help = 'model weights landmark', default='../../float/trainval.caffemodel')
    parser.add_argument('--testImgList', type = str, help = 'test image list', default='testImgList.txt')
    parser.add_argument('--inputImgPath', type = str, help = 'input image path', default='./testImages/')
    parser.add_argument('--outputPath', type = str, help = 'output image path', default='./output')
    parser.add_argument('--gpu', type = int, help = 'select gpu')
    args = parser.parse_args()
    
    args.points = 'fc6_points'
    #args.sex = 'fc6_sex'
    #args.age = 'fc6_age'
    args.im_height = 96
    args.im_width = 72
    args.mean_value = 127.5
    args.scale = 0.00784315

    print 'model define file: %s' % args.model
    print 'model weights file: %s' % args.weights
    print 'gpu id: %d' % args.gpu
    print 'model points output blob name: %s' % args.points
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

    outputPath = args.outputPath

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    getOutput(args)

