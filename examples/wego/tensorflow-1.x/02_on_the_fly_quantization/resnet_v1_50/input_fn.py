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

##############################
# Float Model Configurations #
##############################
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
PREPROCESS_TYPE = "vgg"

###########################
# Quantize Configurations #
###########################
CALIB_BATCH_SIZE = 50
CALIB_IMAGE_DIR = "/scratch/data/Imagenet/val_dataset"
CALIB_IMAGE_LIST = "tf_resnetv1_50_imagenet_224_224_0.38_4.3G_3.0/data/calib_list.txt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + '/tf_resnetv1_50_imagenet_224_224_0.38_4.3G_3.0/code/test')
print(BASE_DIR)
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

preprocess_type = PREPROCESS_TYPE
calib_image_dir = CALIB_IMAGE_DIR
calib_image_list = CALIB_IMAGE_LIST
calib_batch_size = CALIB_BATCH_SIZE
input_height = INPUT_HEIGHT
input_width = INPUT_WIDTH


caliber = Caliber(calib_image_dir, calib_image_list, preprocess_type,
                  input_height, input_width, calib_batch_size)
calib_input = caliber._calib_input
