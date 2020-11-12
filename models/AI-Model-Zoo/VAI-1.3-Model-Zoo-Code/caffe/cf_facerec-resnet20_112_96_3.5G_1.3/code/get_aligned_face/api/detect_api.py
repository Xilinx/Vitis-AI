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

import numpy as np
import scipy.misc
import scipy.io
from matplotlib.patches import Rectangle 
import datetime
import cv2
import sys


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

class Detect(object):
  def __init__(self):
    self.expand_scale_=0.0
    self.force_gray_=False
    self.input_mean_value_=128.0
    self.input_scale_=1.0
    self.pixel_blob_name_='pixel-loss'
    self.bb_blob_name_='bb-output-tiled'
    
    self.res_stride_=4
    self.det_threshold_=0.7
    self.nms_threshold_=0.3
    self.caffe_path_=""
    self.input_channels_=3
  def model_init(self,caffe_python_path,model_path,def_path):
    sys.path.insert(0,caffe_python_path)
    import caffe
    self.caffe_path_=caffe_python_path
    self.net_=caffe.Net(def_path,model_path,caffe.TEST)  
  def detect(self,image):
    #sys.path.insert(0,self.caffe_path_)
    import caffe
    #caffe.set_mode_cpu()
    #caffe.set_device(0)
    self.transformer_=caffe.io.Transformer({'data': (1,self.input_channels_,image.shape[0],image.shape[1])})
    if self.force_gray_:
      image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      self.input_channels_=1
    else:
      self.transformer_.set_transpose('data', (2,0,1))
    transformed_image=self.transformer_.preprocess('data',image)
    transformed_image=(transformed_image-self.input_mean_value_)*self.input_scale_
    sz=image.shape
    self.net_.blobs['data'].reshape(1, self.input_channels_, sz[0], sz[1])
    self.net_.blobs['data'].data[0, ...] = transformed_image
    output = self.net_.forward()
    prob = output[self.pixel_blob_name_][0, 1, ...]
    bb = output[self.bb_blob_name_][0, ...]
    gy = np.arange(0, sz[0], self.res_stride_)
    gx = np.arange(0, sz[1], self.res_stride_)
    gy = gy[0 : bb.shape[1]]
    gx = gx[0 : bb.shape[2]]
    [x, y] = np.meshgrid(gx, gy)
    
    #print bb.shape[1],len(gy),sz[0],sz[1]
    bb[0, :, :] += x
    bb[2, :, :] += x
    bb[1, :, :] += y
    bb[3, :, :] += y
    bb = np.reshape(bb, (4, -1)).T
    prob = np.reshape(prob, (-1, 1))
    bb = bb[prob.ravel() > self.det_threshold_, :]
    prob = prob[prob.ravel() > self.det_threshold_, :]
    rects = np.hstack((bb, prob))
    keep = nms(rects, self.nms_threshold_)	
    rects = rects[keep, :]
    rects_expand=[]
    for rect in rects:
      rect_expand=[]
      rect_w=rect[2]-rect[0]
      rect_h=rect[3]-rect[1]
      rect_expand.append(int(max(0,rect[0]-rect_w*self.expand_scale_)))
      rect_expand.append(int(max(0,rect[1]-rect_h*self.expand_scale_)))
      rect_expand.append(int(min(sz[1],rect[2]+rect_w*self.expand_scale_)))
      rect_expand.append(int(min(sz[0],rect[3]+rect_h*self.expand_scale_)))
      rects_expand.append(rect_expand)
 
    return rects_expand 
