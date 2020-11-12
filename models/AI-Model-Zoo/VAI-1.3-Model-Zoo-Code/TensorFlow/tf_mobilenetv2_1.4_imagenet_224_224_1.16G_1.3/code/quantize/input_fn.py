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

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR + '/test')
from eval_tf_classification_models_alone import DataLoader

class Caliber(object):
  def __init__(self,
               calib_image_dir,
               calib_image_list,
               preprocess_type='inception',
               input_height=224,
               input_width=224,
               calib_batch_size=64):
    self.calib_image_dir = calib_image_dir
    self.calib_image_list = calib_image_list
    self.preprocess_type = preprocess_type
    self.input_height = input_height
    self.input_width = input_width
    self.calib_batch_size = calib_batch_size

  def _calib_input(self, iter):
    with tf.Session() as sess:
      images = []
      data_loader = DataLoader(self.input_height, self.input_width)
      image, input_plhd = data_loader.build_preprocess(
        style=self.preprocess_type)
      lines = open(self.calib_image_list).readlines()
      for index in range(0, self.calib_batch_size):
        curline = lines[iter * self.calib_batch_size + index]
        calib_image_name = curline.strip()
        img_path = os.path.join(self.calib_image_dir, calib_image_name)
        image_calib = sess.run(image, feed_dict={input_plhd: img_path})
        image_calib = np.squeeze(image_calib)
        images.append(image_calib)
      return {"input": images}


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


preprocess_type = get_config(key="PREPROCESS_TYPE", default_value="inception")
calib_image_dir = get_config(key="CALIB_IMAGE_DIR",
                             default_value="../../data/Imagenet/val_dataset")
calib_image_list = get_config(key="CALIB_IMAGE_LIST",
                              default_value="../../data/calib_list.txt")
calib_batch_size = int(get_config(key="CALIB_BATCH_SIZE", default_value=50))
input_height = int(get_config(key="INPUT_HEIGHT", default_value=224))
input_width = int(get_config(key="INPUT_WIDTH", default_value=224))


caliber = Caliber(calib_image_dir, calib_image_list, preprocess_type,
                  input_height, input_width, calib_batch_size)
calib_input = caliber._calib_input
