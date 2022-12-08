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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import cv2
import numpy as np
keras=tf.keras

def get_images_infor_from_file(image_dir, size):
  cnt = 0
  listimage = os.listdir(image_dir)
  imgs = []
  for i in range(len(listimage)):
    img_name  = listimage[i]
    img_path = os.path.join(image_dir, img_name)
    imgs.append(img_path)
    cnt = cnt+1
    if cnt == size:
      break
  return imgs


class ImagenetSequence(Sequence):
  def __init__(self, filenames, batch_size):
    self.filenames = filenames
    self.batch_size = batch_size

  def __len__(self):
    return int(np.ceil(len(self.filenames) / float(self.batch_size)))

  def __getitem__(self, idx):
    batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

    processed_imgs = []
    self.central_fraction = 0.875
    for img_path in batch_x:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image,tf.float32)
        image = tf.image.central_crop(image, central_fraction=self.central_fraction)
    
        image = tf.expand_dims(image, 0)
        image = tf.compat.v1.image.resize_bilinear(image,[224,224])
        image = tf.subtract(image, 127.5)
        #image = tf.multiply(image, 2.0)
        image = tf.multiply(image, 0.0078431)
        processed_imgs.append(image)
    return keras.layers.Concatenate(0)(processed_imgs)

