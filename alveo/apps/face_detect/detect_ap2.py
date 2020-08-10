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
import sys

import numpy as np
import nms
import cv2
#import time

import detect_util
import math

def padProcess(image, in_w, in_h):
    oriSize = image.shape
    sz_ratio = in_w/float(in_h)
    if oriSize[1] / float(oriSize[0]) >= sz_ratio:
        newHeight = int(math.ceil(oriSize[1]/sz_ratio))
        imagePad = np.zeros((newHeight, oriSize[1], 3), np.uint8)
    else:
        newWidth = int(math.ceil(oriSize[0]*sz_ratio))
        imagePad = np.zeros((oriSize[0], newWidth, 3), np.uint8)

    imagePad[0:oriSize[0], 0:oriSize[1], :] = image
    return imagePad

# Imagepad,resize, simple HWC->CHW and mean subtraction/scaling
# returns tensor ready for fpga execute

def det_preprocess(image_ori, resize_w, resize_h, dest):
    input_scale_ = 1.0
    input_mean_value_ = 128
    imagePad = padProcess(image_ori, resize_w, resize_h)
    image = cv2.resize(imagePad,(resize_w, resize_h), interpolation = cv2.INTER_CUBIC)
    image = image.astype(np.float)
    szs = (float(imagePad.shape[0])/float(image.shape[0]), float(imagePad.shape[1])/float(image.shape[1]))
    dest[:] = np.transpose(image,(2,0,1))
    dest -= input_mean_value_
    dest *= input_scale_
    dest = np.expand_dims(dest,0)
    return  szs

   

# takes dict of two outputs from XDNN, pixel-conv and bb-output
# returns bounding boxes
def det_postprocess(pixel_conv, bb, sz, szs):
    res_stride_=4
    det_threshold_=0.7
    nms_threshold_=0.3
    expand_scale_=0.0
    pixel_conv_tiled = detect_util.GSTilingLayer_forward(pixel_conv, 8)
    prob = detect_util.SoftmaxLayer_forward(pixel_conv_tiled)
    bb = detect_util.GSTilingLayer_forward(bb, 8)
    prob = prob[0,1,...]
    bb = bb[0, ...]
    gx_shape = bb.shape[2]*res_stride_
    gy_shape = bb.shape[1]*res_stride_
    gy = np.arange(0, gy_shape, res_stride_)
    gx = np.arange(0, gx_shape, res_stride_)
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
    bb = bb[prob.ravel() > det_threshold_, :]
    prob = prob[prob.ravel() > det_threshold_, :]
    rects = np.hstack((bb, prob))
    keep = nms.nms(rects, nms_threshold_)	
    rects = rects[keep, :]
    return rects

