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

"""Test a regression network on ai challenger."""

import time
import math

import os.path as osp
import sys
import argparse


import numpy as np
import cv2

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)




parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--data', help='anno image path')
parser.add_argument('--caffe', help='load a model for training or evaluation')
parser.add_argument('--cpu', action='store_true', help='CPU ONLY')
parser.add_argument('--weights', help='weights path')
parser.add_argument('--model', help='model path')
parser.add_argument('--anno', help='anno file path')
parser.add_argument('--output', default='result/', help='output_path')
parser.add_argument('--name', help='output name, default eval', default='eval')
parser.add_argument('--input', help='input name in the first layer', default='image')
parser.add_argument('--width', default=128, type=int, help='width of input image')
parser.add_argument('--height', default=224, type=int, help='height of input image')
args = parser.parse_args()

# Add caffe to PYTHONPATH
caffe_path = osp.join(args.caffe, 'python')
add_path(caffe_path)

import caffe

class Config:
    def __init__(self):
        self.use_gpu = not args.cpu
        self.gpuID = args.gpu
        self.caffemodel = args.weights 
        self.deployFile = args.model
        self.description_short = 'googlenet_regression'
        self.width = args.width #128
        self.height = args.height #224
        self.npoints = 14
        self.mean = [104, 117, 123]
        self.result_dir = args.output 
        self.test_image_dir = args.data 
        self.test_anno_file = args.anno 
        # 1: R_shoulder, 2: R_elbow, 3: R_wrist, 4: L_shoulder, 5: L_elbow, 6: L_wrist, 7: R_hip,
        # 8: R_knee, 9: R_ankle, 10: L_hip, 11: L_knee, 12: L_ankle, 13: head, 14: neck


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def preprocess(img, param):
    img_out = cv2.resize(img, (param.width, param.height))
    img_out = np.float32(img_out)
    img_out[:, :, 0] = img_out[:, :, 0] - param.mean[0]
    img_out[:, :, 1] = img_out[:, :, 1] - param.mean[1]
    img_out[:, :, 2] = img_out[:, :, 2] - param.mean[2]
    # change H*W*C -> C*H*W
    return np.transpose(img_out, (2, 0, 1))


def applymodel(net, image, param):
    # Select parameters from param
    width = param.width
    height = param.height
    npoints = param.npoints

    imageToTest = preprocess(image, param)
    net.blobs['data'].data[...] = imageToTest.reshape((1, 3, height, width))
    net.forward()
    prediction = net.blobs['pred_coordinate'].data[0]
    pred_cooridnate = np.zeros((param.npoints, 2), dtype=np.float)
    for j in range(param.npoints):
        pred_cooridnate[j, 0] = prediction[j * 2]
        pred_cooridnate[j, 1] = prediction[j * 2 + 1]
    #pred_visible = net.blobs['pred_visible'].data[0]
    pred_visible = None
    return pred_cooridnate, pred_visible


def draw_joints_16(test_image, pred_coordinate, save_image, param):
    image = cv2.imread(test_image)
    joints = np.zeros(pred_coordinate.shape, dtype=np.int)
    for j in range(pred_coordinate.shape[0]):
        joints[j, 0] = int(round(pred_coordinate[j, 0] * image.shape[1] / param.width))
        joints[j, 1] = int(round(pred_coordinate[j, 1] * image.shape[0] / param.height))

    # draw joints in green spots
    for j in range(len(joints)):
        cv2.circle(image, (joints[j, 0], joints[j, 1]), 5, (0, 255, 0), 2)
    # draw torso in yellow lines
    torso = [[0, 6], [6, 9], [0, 13], [3, 13], [3, 9], [12, 13]]
    for item in torso:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (0, 0, 0), 2)
    # draw left part in pink lines
    lpart = [[3, 4], [4, 5], [9, 10], [10, 11]]
    for item in lpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 0), 2)
    # draw right part in blue lines
    rpart = [[0, 1], [1, 2], [6, 7], [7, 8]]
    for item in rpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (0, 0, 255), 2)
    cv2.imwrite(save_image, image)


if __name__ == '__main__':
    param = Config()
    if param.use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(param.gpuID)
    net = caffe.Net(param.deployFile, param.caffemodel, caffe.TEST)
    net.name = param.description_short

    # test a folder
    test_image_dir = param.test_image_dir
    test_anno_file = open(param.test_anno_file, 'r')
    result_dir = param.result_dir

    lines = test_anno_file.readlines()
    precision = np.zeros(param.npoints)
    number = 0
    for i in range(len(lines)):
        print("image number: " + "%d" % i)
        info = lines[i].split("\n")[0].split(" ")
        test_image = test_image_dir + info[0]
        save_image = result_dir + info[0]
        image = cv2.imread(test_image)
        #print(test_image)
        if image is None:
            continue
        if len(image.shape) != 3:
            continue
        joints = [int(round(float(item))) for item in info[1:29]]
        weights = [int(round(float(item))) for item in info[29:57]]
        visible = [int(round(float(item))) for item in info[57:]]
        print(weights[1::2])
        # # draw gt
        # gt_cooridnate = np.zeros((param.npoints, 2), dtype=np.float)
        # for j in range(param.npoints):
        #     gt_cooridnate[j, 0] = joints[j * 2]
        #     gt_cooridnate[j, 1] = joints[j * 2 + 1]
        # draw_joints_16(test_image, gt_cooridnate, 'gt/' + info[0], param)

        pred_coordinate, pred_visible = applymodel(net, image, param)
        #draw_joints_16(test_image, pred_coordinate, save_image, param)
        #if param.npoints == 16:
        #    draw_joints_16(test_image, pred_coordinate, save_image, param)
        px = joints[::2]
        py = joints[1::2]
        threshold = math.sqrt((px[13]-px[12]) ** 2 + (py[13]-py[12]) ** 2)
        temp_precision = np.zeros(param.npoints)
        number += 1
        for j in range(len(px)):
            temp_precision[j] = math.sqrt((px[j]-pred_coordinate[j, 0]) ** 2 + (py[j]-pred_coordinate[j, 1]) ** 2) < 0.5 * threshold
            #if visible[j] == 2:
            if weights[j*2] == 0:
                temp_precision[j] = 1
        precision += temp_precision
        print(temp_precision)
    if number > 0:
        precision /= number
    print(precision)
    print(np.mean(precision))





