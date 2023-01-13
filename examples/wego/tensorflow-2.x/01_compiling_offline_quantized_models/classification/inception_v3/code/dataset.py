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
import cv2
import numpy as np

DEFAULT_IMAGE_SIZE = 299

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
  def __init__(self, filenames, batch_size, output_height=DEFAULT_IMAGE_SIZE, output_width=DEFAULT_IMAGE_SIZE, central_fraction=0.875):
    self.filenames = filenames
    self.batch_size = batch_size
    self.central_fraction = central_fraction
    self.output_height = output_height
    self.output_width = output_width

  def __len__(self):
    return int(np.ceil(len(self.filenames) / float(self.batch_size)))

  def __getitem__(self, idx):
    batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

    processed_imgs = []

    for filename in batch_x:
      # B G R format
      img = cv2.imread(filename)
      height, width = img.shape[0], img.shape[1]
      img = img.astype(float)
      ratio_offset = (1 - self.central_fraction) / 2
      left = int(ratio_offset * width)
      top = int(ratio_offset * height)
      new_height = int(height * self.central_fraction)
      new_width = int(width * self.central_fraction)
      img_crop = img[top:top+new_height, left:left+width, :]
      img_crop = img_crop / 255.0
      resized_img = cv2.resize(img_crop, (self.output_height, self.output_width), interpolation = cv2.INTER_LINEAR)
      image = (resized_img - 0.5) * 2.0
      processed_imgs.append(image[:, :, ::-1])
    return np.array(processed_imgs)
