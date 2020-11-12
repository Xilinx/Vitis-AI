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
import os
import cv2
import numpy as np
import tensorflow as tf


def preprocess(image):
  image = image.astype('float32')
  R_MEAN = 123.68
  G_MEAN = 116.78
  B_MEAN = 103.94
  mean = np.array([B_MEAN, G_MEAN, R_MEAN], dtype=np.float32)
  std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
  image = (image - mean) / std
  return image


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


calib_image_dir = get_config(
    key="CALIB_IMAGE_DIR",
    default_value="../../data/EDD/images/")
calib_image_list = get_config(
    key="CALIB_IMAGE_LIST",
    default_value="../../data/EDD/val_image_list.txt")
calib_batch_size = int(get_config(key="CALIB_BATCH_SIZE", default_value=50))
input_height = int(get_config(key="INPUT_HEIGHT", default_value=320))
input_width = int(get_config(key="INPUT_WIDTH", default_value=320))


def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  with tf.Graph().as_default():
    for index in range(0, calib_batch_size):
      curline = line[iter * calib_batch_size + index]
      calib_image_name = curline.strip()
      image_path = os.path.join(calib_image_dir, calib_image_name + ".jpg")
      image = cv2.imread(image_path)
      image = np.array(cv2.resize(image, (input_height, input_width)))
      image = preprocess(image)
      images.append(image)
  return {"image": images}
