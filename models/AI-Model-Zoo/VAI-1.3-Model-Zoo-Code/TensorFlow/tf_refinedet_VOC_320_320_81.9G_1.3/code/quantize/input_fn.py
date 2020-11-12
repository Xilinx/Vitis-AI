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
import cv2
import numpy as np
import tensorflow as tf

class Caliber(object):
  def __init__(self,
               calib_image_dir,
               calib_image_list,
               input_height=224,
               input_width=224,
               calib_batch_size=64):
    self.calib_image_dir = calib_image_dir
    self.calib_image_list = calib_image_list
    self.input_height = input_height
    self.input_width = input_width
    self.calib_batch_size = calib_batch_size

  def preprocess(self, image):
    image = image.astype('float32')
    R_MEAN = 123.68
    G_MEAN = 116.78
    B_MEAN = 103.94
    mean = np.array([B_MEAN, G_MEAN, R_MEAN], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    image = (image - mean) / std
    return image

  def _calib_input(self, iter):
    images = []
    lines = open(self.calib_image_list).readlines()
    for index in range(0, self.calib_batch_size):
      curline = lines[iter * self.calib_batch_size + index]
      calib_image_name = curline.strip()
      img_path = os.path.join(self.calib_image_dir, calib_image_name)
      image = cv2.imread(img_path)
      image = np.array(cv2.resize(image, (self.input_height, self.input_width)))
      pre_image = self.preprocess(image)
      images.append(pre_image.tolist())
    return {"image": images}


def get_config(key=None, default_value=None):
  if not key:
    raise ValueError("Please assign a key.")
  if not default_value:
    raise ValueEror("Please assign a default_value")

  config = os.environ
  if key in config:
    value = config[key]
    print("Get {} from env: {}".format(key, value))
    return value
  else:
    print("Fail to get {} from env, use default value {}".format(
      key, default_value))
    return default_value


calib_image_dir = get_config(key="CALIB_IMAGE_DIR",
                             default_value="../../data/VOC/images")
calib_image_list = get_config(key="CALIB_IMAGE_LIST",
                              default_value="../../data/calib_list.txt")
calib_batch_size = int(get_config(key="CALIB_BATCH_SIZE", default_value=50))
input_height = int(get_config(key="INPUT_HEIGHT", default_value=320))
input_width = int(get_config(key="INPUT_WIDTH", default_value=320))


caliber = Caliber(calib_image_dir, calib_image_list,
                  input_height, input_width, calib_batch_size)
calib_input = caliber._calib_input
