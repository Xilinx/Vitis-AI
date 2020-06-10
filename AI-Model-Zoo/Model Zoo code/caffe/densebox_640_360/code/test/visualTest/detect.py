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
import math
import time
import argparse
import os
import commands

import sys
sys.path.insert(0,'../../../../../../caffe-xilinx/python')
sys.path.insert(0,'../../../../../../caffe-xilinx/python/caffe')

import caffe
# load argparse

def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def padProcess(image):
    oriSize = image.shape
    sz_ratio = 640/360.0
    if oriSize[1] / float(oriSize[0]) >= sz_ratio:
        newHeight = int(math.ceil(oriSize[1]/sz_ratio))
        imagePad = np.zeros((newHeight, oriSize[1], 3), np.uint8)
    else:
        newWidth = int(math.ceil(oriSize[0]*sz_ratio))
        imagePad = np.zeros((oriSize[0], newWidth, 3), np.uint8)

    imagePad[0:oriSize[0], 0:oriSize[1], :] = image
    return imagePad

def detect(args, testImgList, resolution, threshold, nms_threshold, outputPath):
    for i, line in enumerate(testImgList):
        image_name =  args.inputImgPath + line.strip()
        # generate image data for net, set shape multiplier of 32
        image_ori = cv2.imread(image_name, cv2.IMREAD_COLOR)
        imagePad = padProcess(image_ori) 
        image = cv2.resize(imagePad,(32*(640/32), 32*(360/32)), interpolation = cv2.INTER_CUBIC)
        szs = (float(imagePad.shape[0])/float(image.shape[0]), float(imagePad.shape[1])/float(image.shape[1]))
        sz = image.shape
        image = image.astype(np.float)
        image = image - 128
        image = np.transpose(image, (2, 0, 1))
        # reshape net and load data
        net.blobs['data'].reshape(1, 3, sz[0], sz[1])
        net.blobs['data'].data[...] = image
        # forward and time count
        output = net.forward()
    
        # generate result
        # args
        prob = output[args.score][0, 1, ...]
        bb = output[args.bbox][0, ...]
        gy = np.arange(0, sz[0], resolution)
        gx = np.arange(0, sz[1], resolution)
        [x, y] = np.meshgrid(gx, gy)
        bb[0, :, :] = bb[0, :, :] + x
        bb[0, :, :] = bb[0, :, :] * szs[1]
        bb[1, :, :] = bb[1, :, :] + y
        bb[1, :, :] = bb[1, :, :] * szs[0]
        bb[2, :, :] = bb[2, :, :] + x
        bb[2, :, :] = bb[2, :, :] * szs[1]
        bb[3, :, :] = bb[3, :, :] + y
        bb[3, :, :] = bb[3, :, :] * szs[0]
        bb = np.reshape(bb, (4, -1)).T
        prob = np.reshape(prob, (-1, 1))
        bb = bb[prob.ravel() > threshold, :]
        prob = prob[prob.ravel() > threshold, :]
        # nms
        rects = np.hstack((bb, prob))
        keep = nms(rects, nms_threshold)
        rects = rects[keep, :]
        # write result to file
        for rect in rects:
            cv2.rectangle(image_ori,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),3)

        outputImgName = args.outputPath + 'output_' + image_name.split('/')[-1]

        cv2.imwrite(outputImgName, image_ori)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'analysis densebox model, must be deploy, not train')
    parser.add_argument('--model', help = 'model structure densebox', default='../../../float/test.prototxt')
    parser.add_argument('--weights', help = 'model weights densebox', default='../../../float/trainval.caffemodel')
    parser.add_argument('--testImgList', type = str, help = 'test image list', default='image_test_list.txt')
    parser.add_argument('--inputImgPath', type = str, help = 'input image path', default='./testImages/')
    parser.add_argument('--outputPath', type = str, help = 'output image path', default='./output')
    parser.add_argument('--gpu', type = int, help = 'select gpu')
    parser.add_argument('--score', type = int, choices = [0, 1], default = 0,
                        help = 'score output blob name in model structure, 0 for pixel-loss, 1 for score_softmax[default = 1]')
    parser.add_argument('--bbox', type = int, choices = [0, 1], default = 0,
                        help = 'bbox output blob name in model structure, 0 for bb-output-tiled, 1 for bbox_output[default = 1]')

    args = parser.parse_args()

    assert os.path.exists(args.model)
    assert os.path.exists(args.weights)
    args.score = ['pixel-loss', 'score_softmax'][args.score]#Keep the key value consistent with the top name of the softmax layer of *.prototxt file
    args.bbox = ['bb-output-tiled', 'bbox_output'][args.bbox]#Keep the key value consistent with the top name of the GSTiling layer of *.prototxt file

    work_dir = os.getcwd() + '/'

    print('model define file: %s'%(args.model))
    print('model weights file: %s'%(args.weights))
    print('gpu id: %d'%(args.gpu))
    print('model score output blob name: %s'%(args.score))
    print('model bbox output blob name: %s'%(args.bbox))

    # parameter
    resolution = 4
    threshold = 0.9
    nms_threshold = 0.3

    # load caffe
    os.environ['GLOG_minloglevel'] = '3'

    # load network
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    model_def = args.model
    model_weights = args.weights
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # read FDDB image list
    image_list_file = open(args.testImgList, 'r')
    image_list = image_list_file.readlines()
    image_list_file.close()
    
    outputPath = args.outputPath

    if not os.path.exists(outputPath):
            os.makedirs(outputPath)
    detect(args, image_list, resolution, threshold, nms_threshold, outputPath)

