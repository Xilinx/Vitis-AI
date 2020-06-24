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

import tensorflow as tf

def preprocess_v1(image, new_height, new_width):
  ''' Resize and normalize the image
  Args:
      image: image of shape [height, width, 3] with RGB channel order
             pixel value in [0.0, 255.0]
  '''
  resized_image = tf.image.resize_images(image, tf.stack([new_height, new_width]),
                      method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
  preprocessed_image = (2.0 / 255.0) * resized_image - 1.0
  return preprocessed_image

def preprocess_v2(image, new_height, new_width):
  ''' Resize and normalize the image
  Args:
      image: image of shape [height, width, 3] with RGB channel order
             pixel value in [0.0, 255.0]
  '''
  resized_image = tf.image.resize_images(image, tf.stack([new_height, new_width]),
                      method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
  if resized_image.shape.as_list()[3] == 3:
    channel_means = [123.68, 116.779, 103.939]
    return resized_image - [[channel_means]]
  else:
    return resized_image


# key: feature_extractor.type
PREPROCESS_FUNC = {'ssd_inception_v2':  preprocess_v1,
                   'ssd_mobilenet_v1': preprocess_v1,
                   'ssd_mobilenet_v2': preprocess_v1,
                   'ssd_mobilenet_v1_fpn': preprocess_v1,
                   'ssd_resnet50_v1_fpn':  preprocess_v2}
