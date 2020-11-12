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

from PIL import Image


def letterbox_image(image, size):
  '''resize image with unchanged aspect ratio using padding'''
  iw, ih = image.size
  w, h = size
  scale = min(w / iw, h / ih)
  nw = int(iw * scale)
  nh = int(ih * scale)

  image = image.resize((nw, nh), Image.BICUBIC)
  new_image = Image.new('RGB', size, (128, 128, 128))
  new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
  return new_image


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
    default_value="../../data/voc2007_test/images")
calib_image_list = get_config(
    key="CALIB_IMAGE_LIST",
    default_value="../../data/voc2007_test/images/test.txt")
calib_batch_size = int(get_config(key="CALIB_BATCH_SIZE", default_value=1))
input_height = int(get_config(key="INPUT_HEIGHT", default_value=416))
input_width = int(get_config(key="INPUT_WIDTH", default_value=416))


def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  with tf.Graph().as_default():
    for index in range(0, calib_batch_size):
      curline = line[iter * calib_batch_size + index]
      calib_image_name = curline.strip()
      image_path = os.path.join(calib_image_dir, calib_image_name)
      image = Image.open(image_path)

      model_image_size = (input_height, input_width)
      if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
      else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
      image_data = np.array(boxed_image, dtype='float32')
      image_data /= 255.

      images.append(image_data)
  return {"input_1": images}
