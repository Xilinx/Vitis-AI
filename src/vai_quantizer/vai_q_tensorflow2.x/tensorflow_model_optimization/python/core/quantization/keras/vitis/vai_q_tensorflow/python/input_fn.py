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

import cv2
import os
import sys
import types

import numpy as np
import tensorflow as tf


def check_images(image_dir, image_list, iterations, batch_size):
  """Check images validation"""
  if not tf.gfile.Exists(image_list):
    raise ValueError("Cannot find image_list file {}.".format(image_list))
  text = open(image_list).readlines()
  print(
      "Total images for calibration: {}\ncalib_iter: {}\nbatch_size: {}".format(
          len(text), iterations, batch_size))
  if (len(text) < iterations * batch_size):
    raise RuntimeError(
        "calib_iter * batch_size > number of images, please decrease calib_iter or batch_size"
    )


def convert_bgr_to_rgb(image):
  """Convert BGR to RGB"""
  B, G, R = cv2.split(image)
  return cv2.merge([R, G, B])


def means_subtraction(image, means):
  """Subtract image means for RGB channels"""
  B, G, R = cv2.split(image)
  R = R - means[0]
  G = G - means[1]
  B = B - means[2]
  return cv2.merge([B, G, R])


def scale_image(image, scales):
  """scale image, often used to normalize image"""
  B, G, R = cv2.split(image)
  B = B * scales[0]
  G = G * scales[1]
  R = R * scales[2]
  return cv2.merge([B, G, R])


def central_crop(image, crop_height, crop_width):
  """Central crop image"""
  image_height = image.shape[0]
  image_width = image.shape[1]

  offset_height = (image_height - crop_height) // 2
  offset_width = (image_width - crop_width) // 2

  return image[offset_height:offset_height +
               crop_height, offset_width:offset_width + crop_width]


def resize(image, image_height, image_width):
  """Resize image"""
  return cv2.resize(
      image, (image_height, image_width), interpolation=cv2.INTER_NEAREST)


def nomalize_image(image):
  """Nomalize image from [0,255] to "[-1, 1]"""
  image = image / 255.0
  image = 2 * (image - 0.5)
  return image


def gen_imagenet_input_fn(input_node, image_dir, image_list, calib_iter,
                          batch_size, image_height, image_width, size_type,
                          means, scales, normalize):
  """Generate imagenet input_fn"""

  check_images(image_dir, image_list, calib_iter, batch_size)

  def imagenet_input_fn(iter):
    """imagenet input function to load image and do preprocessing for quantize calibraton,
        as the calibration process do not need labels, the return value only contain
        images without labels"""
    if len(input_nodes) != 1:
      raise ValueError(
          "Default input_fn only support single input network, but {} found.".
          format(len(input_nodes)))

    text = open(image_list).readlines()
    images = []
    for i in range(0, batch_size):
      image_name = text[iter + i].split(' ')[0]
      image_file = os.path.join(image_dir, image_name.strip())
      if not tf.gfile.Exists(image_file):
        raise ValueError("Cannot find image file {}.".format(image_file))
      image = cv2.imread(image_file)
      if size_type == 0:
        image = central_crop(image, image_height, image_width)
      elif size_type == 1:
        image = resize(image, image_height, image_width)
      else:
        raise ValueError("Invalid size_type")
      image = means_subtraction(image, means)
      if scales != 1:
        image = scale_image(image, scales)
      if normalize:
        image = nomalize_image(image)
      image = convert_bgr_to_rgb(image)
      images.append(image)
    return {input_nodes[0]: images}

  return default_input_fn


def gen_random_input_fn(input_nodes, input_shapes, input_dtypes):
  """Generate random input_fn"""

  def random_input_fn(iter):
    feed_dict = dict()
    for input_node, input_shape, input_dtype in zip(input_nodes, input_shapes,
                                                    input_dtypes):
      in_shape = input_shape.copy()
      if input_shape[0] is None or input_shape[0] == -1:
        in_shape[0] = 1

      dtype_range = {
          np.bool_: (False, True),
          np.bool8: (False, True),
          np.uint8: (0, 255),
          np.uint16: (0, 65535),
          np.int8: (-128, 127),
          np.int16: (-32768, 32767),
          np.int64: (-2**63, 2**63 - 1),
          np.uint64: (0, 2**64 - 1),
          np.int32: (-2**31, 2**31 - 1),
          np.uint32: (0, 2**32 - 1),
          np.float32: (-1, 1),
          np.float64: (-1, 1)
      }
      min_value = dtype_range[np.dtype(input_dtype).type][0]
      max_value = dtype_range[np.dtype(input_dtype).type][1]
      feed_dict[input_node] = np.random.random(in_shape) * (
          max_value - min_value) + min_value
    return feed_dict

  return random_input_fn
