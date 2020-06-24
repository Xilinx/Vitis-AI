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
import sys
import cv2
import time


class LandmarkPredict(object):
  def __init__(self):
    self.caffe_python_path=""
    self.deploy_path=""
    self.caffemodel_path=""
    self.input_width=72
    self.input_height=96
    self.mean_value=127.5
    self.scale=0.00784315
    self.landmark_blob_name="fc6_points"
    self.caffe_path=""
  def model_init(self,caffe_python_path,caffemodel_path,deploy_path,input_height,input_width):
    self.caffe_python_path=caffe_python_path
    sys.path.insert(0,caffe_python_path)
    import caffe
    self.caffe_path_=caffe_python_path
    self.input_width=input_width
    self.input_height=input_height
    self.net=caffe.Net(deploy_path,caffemodel_path,caffe.TEST)

  def predict(self,image,bbox):
    #sys.path.insert(0,self.caffe_python_path)
    import caffe
    h, w, _ = image.shape

    # every bbox mean a face, should be expanded
    xmin = max(0, int(bbox[0]))
    xmax = min(w, int(bbox[2]))
    ymin = max(0, int(bbox[1]))
    ymax = min(h, int(bbox[3]))
    crop_img = image[ymin:ymax, xmin:xmax].copy()
    resize_img = cv2.resize(crop_img, (self.input_width, self.input_height))

    # generate image data for input
    input_image = resize_img.astype(np.float)
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = input_image - self.mean_value
    input_image = input_image * self.scale
    
    # load data
    self.net.blobs['data'].data[...] = input_image
    
    output = self.net.forward()
    predict_landmark=output[self.landmark_blob_name][0]

    predict_landmark_face=predict_landmark.copy()
    landmark_re = []
    for k in range(5):
      landmark_re.append(int(xmin + predict_landmark_face[k] * (xmax-xmin)))
      landmark_re.append(int(ymin + predict_landmark_face[k + 5] * (ymax-ymin)))


    return landmark_re
