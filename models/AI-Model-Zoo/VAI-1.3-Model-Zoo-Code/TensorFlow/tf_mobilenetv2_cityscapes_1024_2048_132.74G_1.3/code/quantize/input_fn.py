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
  """input image process: image padding."""
  IMAGE_PAD_PIXEL = 127.5
  TARGET_H =1024
  TARGET_W = 2048

  image = image.astype('float32')
  orig_h, orig_w = image.shape[:2]

  pad_h = max(TARGET_H - orig_h, 0)
  pad_w = max(TARGET_W - orig_w, 0)

  pad_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=IMAGE_PAD_PIXEL)
  pad_image = 2.0 / 255.0 * pad_image - 1.0
  return pad_image


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
    default_value="../../data/cityscapes/leftImg8bit/val")
calib_image_list = get_config(
    key="CALIB_IMAGE_LIST",
    default_value="../../data/cityscapes/calib.txt")
calib_batch_size = int(get_config(key="CALIB_BATCH_SIZE", default_value=2))
input_height = int(get_config(key="INPUT_HEIGHT", default_value=1024))
input_width = int(get_config(key="INPUT_WIDTH", default_value=2048))


def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  with tf.Graph().as_default():
    for index in range(0, calib_batch_size):
      curline = line[iter * calib_batch_size + index]
      calib_image_name = curline.strip().split()[0]
      image_path = os.path.join(calib_image_dir, calib_image_name)
      image = cv2.imread(image_path)
      image = np.array(image)
      image = preprocess(image)
      images.append(image)
  return {"ImageTensor": images}
