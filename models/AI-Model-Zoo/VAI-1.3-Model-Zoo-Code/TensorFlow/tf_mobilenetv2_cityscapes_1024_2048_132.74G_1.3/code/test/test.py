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

"""Visualization code for semantic segmentation on Cityscapes with Deeplabv3+(MobileNetV2)."""
#@title Imports

import os
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import argparse
import glob
#tf.device('/gpu:1')

class Configs():
    def __init__(self):

        parser = argparse.ArgumentParser("Deeplabv3+_MobileNetV2")
        #dataset
        parser.add_argument('--dataset', type=str, default='cityscapes', help='dataset name')
        parser.add_argument('--data_folder', type=str, default='./data/cityscapes/val', help='path to dataset')
        parser.add_argument('--nclass', type=int, default=19, help='class numbers')
        parser.add_argument('--target_h', type=int, default=1024, help='origianl image height')
        parser.add_argument('--target_w', type=int, default=2048, help='origianl image width')
        parser.add_argument('--image_pad_pixel', type=float, default=127.5, help='pixel value for image padding')
        #weight
        parser.add_argument('--pb_file', type=str, default='./loat/final_1024x2048.pb', help='pb file path')
        parser.add_argument('-input_node', type=str, default='ImageTensor:0', help='input node name')
        parser.add_argument('-output_node', type=str, default='ResizeBilinear_3:0', help='output node name')

        #demo options
        parser.add_argument('--gray2color', action='store_true', help='True for convert gray prediction to color')
        parser.add_argument('--savedir', type=str, default='./data/demo', help='pth for save prediction')

        #quantize
        parser.add_argument('--gpus', type=str, default='0', help='gpu id')
        parser.add_argument('--use_quantize', type=bool, default=False, help='if evaluate quantized model')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

def Gray_to_Color(img):
    """ color map for cityscapes"""
    label_to_color = {
        0: [128, 64,128],
        1: [232, 35,244],
        2: [ 70, 70, 70],
        3: [156,102,102],
        4: [153,153,190],
        5: [153,153,153],
        6: [30,170, 250],
        7: [0,220, 220],
        8: [35,142, 107],
        9: [152,251,152],
        10: [180,130,70],
        11: [60, 20, 220],
        12: [0,  0,  255],
        13: [ 142,  0, 0],
        14: [70,  0, 0],
        15: [100, 60,0],
        16: [100, 80,0],
        17: [230,  0, 0],
        18: [32, 11, 119],
        19: [81,  0, 81],
        255: [0, 0, 0]}
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])
    return img_color

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""
  '''
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'ResizeBilinear_3:0'
  FROZEN_GRAPH = 'final_model.pb'
  TARGET_H, TARGET_W = 1025, 2049
  IMAGE_PAD_PIXEL = 127.5
  '''
  def __init__(self, params):
    """Creates and loads pretrained deeplab model."""
    self.WEIGHT = params.pb_file
    self.INPUT_TENSOR_NAME = params.input_node
    self.OUTPUT_TENSOR_NAME = params.output_node
    self.TARGET_H, self.TARGET_W = params.target_h, params.target_w
    self.IMAGE_PAD_PIXEL = params.image_pad_pixel

    self.graph = tf.Graph()
    graph_def = None
    file_path = self.WEIGHT
    f = open(file_path, "rb")
    graph_def = tf.GraphDef.FromString(f.read())
    if graph_def is None:
      raise RuntimeError('Cannot find inference graph.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
    self.sess = tf.Session(graph=self.graph)

  def input_fn(self, image):
    """input image process: image padding."""
    orig_w, orig_h = image.size
    target_w, target_h = self.TARGET_W, self.TARGET_H
    pad_h = max(target_h - orig_h, 0)
    pad_w = max(target_w - orig_w, 0)

    pad_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=self.IMAGE_PAD_PIXEL)
    pad_image = 2.0 / 255.0 * pad_image - 1.0
    return pad_image

  def output_fn(self, image, batch_seg_map):
    """output process on batch_seg_map: [1, H, W, C]"""

    batch_seg_map_argmax= np.argmax(batch_seg_map, 3)
    batch_seg_map_argmax_slice = batch_seg_map_argmax[:,0:self.TARGET_H, 0:self.TARGET_W]
    seg_map = batch_seg_map_argmax_slice[0]

    orig_w, orig_h = image.size
    seg_map = seg_map[0:orig_h,0:orig_w]

    return seg_map

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      image: original input image.
      seg_map: Segmentation map.
    """
    pad_image = self.input_fn(image)

    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [pad_image]})

    seg_map = self.output_fn(image, batch_seg_map)

    return seg_map


def segment_single_image(data_root, image_name, save_folder, gray2color):
  """Inferences DeepLab model and visualizes result.
  Args:
       data_folder: test images folder path
       image_name: image name
       save_folde: results save folder path (both for gray ones for mIOU evaluation, and color ones)
       gray_to_color: wether convert gray to color (default: false)
  """
  try:
    jpeg_str = os.path.join(data_root,image_name)
    orignal_im = Image.open(jpeg_str)
  except IOError:
    print('Cannot retrieve image.')
    return

  seg_map = MODEL.run(orignal_im)

  if not os.path.isdir(save_folder):
      os.makedirs(save_folder)

  cv2.imwrite(os.path.join(save_folder, image_name), seg_map)

  if gray2color:
      color_seg_map = Gray_to_Color(seg_map).astype(np.uint8)
      cv2.imwrite(os.path.join(save_folder, 'color_'+image_name), color_seg_map)


if __name__ == '__main__':
  param = Configs().parse()
  os.environ['CUDA_VISIBLE_DEVICES'] = param.gpus
  if param.use_quantize:
      from tensorflow.contrib import decent_q
  MODEL = DeepLabModel(param)
  count = 0
  for city in os.listdir(param.data_folder):
      for image_name in os.listdir(os.path.join(param.data_folder,city)):
          print("image number: " + "%d" % count)
          count += 1
          segment_single_image(os.path.join(param.data_folder,city), image_name, param.savedir, param.gray2color)
