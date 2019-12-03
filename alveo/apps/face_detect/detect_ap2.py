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
import numpy as np

import nms
import time

import detect_util


# simple HWC->CHW and mean subtraction/scaling
# returns tensor ready for fpga execute
def det_preprocess(image, dest):
    input_mean_value_=128.0
    input_scale_=1.0

    # transpose HWC (0,1,2) to CHW (2,0,1)
    dest[:] = np.transpose(image,(2,0,1))
    dest -= input_mean_value_
    dest *= input_scale_
    dest = np.expand_dims(dest,0)


# takes dict of two outputs from XDNN, pixel-conv and bb-output
# returns bounding boxes
def det_postprocess(pixel_conv, bb, sz):
    res_stride_=4
    det_threshold_=0.7
    nms_threshold_=0.3
    expand_scale_=0.0

#    sz=image.shape

    start_time = time.time()

    # Put CPU layers into postprocess
    pixel_conv_tiled = detect_util.GSTilingLayer_forward(pixel_conv, 8)
    prob = detect_util.SoftmaxLayer_forward(pixel_conv_tiled)
    prob = prob[0,1,...]

    bb = detect_util.GSTilingLayer_forward(bb, 8)
    bb = bb[0, ...]

    end_time = time.time()
#    print('detect post-processing time: {0} seconds'.format(end_time - start_time))

    gy = np.arange(0, sz[0], res_stride_)
    gx = np.arange(0, sz[1], res_stride_)
    gy = gy[0 : bb.shape[1]]
    gx = gx[0 : bb.shape[2]]
    [x, y] = np.meshgrid(gx, gy)
    bb[0, :, :] += x
    bb[2, :, :] += x
    bb[1, :, :] += y
    bb[3, :, :] += y
    bb = np.reshape(bb, (4, -1)).T
    prob = np.reshape(prob, (-1, 1))
    bb = bb[prob.ravel() > det_threshold_, :]
    prob = prob[prob.ravel() > det_threshold_, :]
    rects = np.hstack((bb, prob))
    keep = nms.nms(rects, nms_threshold_)
    rects = rects[keep, :]
    rects_expand=[]
    for rect in rects:
      rect_expand=[]
      rect_w=rect[2]-rect[0]
      rect_h=rect[3]-rect[1]
      rect_expand.append(int(max(0,rect[0]-rect_w*expand_scale_)))
      rect_expand.append(int(max(0,rect[1]-rect_h*expand_scale_)))
      rect_expand.append(int(min(sz[1],rect[2]+rect_w*expand_scale_)))
      rect_expand.append(int(min(sz[0],rect[3]+rect_h*expand_scale_)))
      rects_expand.append(rect_expand)
    return rects_expand
