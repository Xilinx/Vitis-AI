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

from tensorflow.keras.utils import Sequence
import cv2
import numpy as np

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

    for filename in batch_x:
      # B G R format
      img = cv2.imread(filename)
      height, width = img.shape[0], img.shape[1]
      img = img.astype(float)

      # aspect_preserving_resize
      smaller_dim = np.min([height, width])
      _RESIZE_MIN = 256
      scale_ratio = _RESIZE_MIN*1.0 / (smaller_dim*1.0)
      new_height = int(height * scale_ratio)
      new_width = int(width * scale_ratio)
      resized_img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR )

      # central_crop
      crop_height = 224
      crop_width = 224
      amount_to_be_cropped_h = (new_height - crop_height)
      crop_top = amount_to_be_cropped_h // 2
      amount_to_be_cropped_w = (new_width - crop_width)
      crop_left = amount_to_be_cropped_w // 2
      cropped_img = resized_img[crop_top:crop_top+crop_height,
              crop_left:crop_left+crop_width, :]

      # sub mean
      _R_MEAN = 123.68
      _G_MEAN = 116.78
      _B_MEAN = 103.94
      _CHANNEL_MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
      means = np.expand_dims(np.expand_dims(_CHANNEL_MEANS, 0), 0)
      meaned_img = cropped_img - means

      # model.predict(np.expand_dims(meaned_img, 0))
      # model.evaluate(np.expand_dims(meaned_img, 0), np.expand_dims(labels[0], 0))
      processed_imgs.append(meaned_img)
      #print(processed_imgs[0].shape)
    #return np.array(processed_imgs)#, np.array(batch_y)
    return np.array(processed_imgs)
