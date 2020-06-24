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

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join('caffe_root/', 'python')
add_path(caffe_path)

import caffe
import numpy as np
import cv2


class Config:
    def __init__(self):
        self.use_gpu = 0
        self.gpuID = 3
        self.caffemodel = osp.join(this_dir, 'model_dir/', 'float', 'trainval.caffemodel')
        self.deployFile = osp.join(this_dir, '../..', 'float', 'test.prototxt')
        self.description_short = 'googlenet_regression'
        self.width = 128
        self.height = 224
        self.npoints = 14
        self.mean = [104, 117, 123]
        self.result_dir = 'result/'
        self.test_image = 'demo/demo.png'
        self.save_image = 'output/demo_output1.png'
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
    pred_visible=None
    return pred_cooridnate, pred_visible


def draw_joints(test_image, pred_coordinate, save_image, param):
    image = cv2.imread(test_image)
    joints = np.zeros(pred_coordinate.shape, dtype=np.int)
    for j in range(pred_coordinate.shape[0]):
        joints[j, 0] = int(round(pred_coordinate[j, 0] * image.shape[1] / param.width))
        joints[j, 1] = int(round(pred_coordinate[j, 1] * image.shape[0] / param.height))

    # draw joints in green spots
    for j in range(len(joints)):
        cv2.circle(image, (joints[j, 0], joints[j, 1]), 5, (0, 255, 0), 2)
    print('write image at:' + save_image)    
    cv2.imwrite(save_image, image)

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

    
    test_image = param.test_image
    save_image = param.save_image
    image = cv2.imread(test_image)
    #print(test_image)
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
    #print(pred_coordinate)
    draw_joints(test_image, pred_coordinate, save_image, param)





