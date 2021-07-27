#!/usr/bin/env python
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import os
import os.path as osp
import cv2


IMAGEROOT = osp.join(os.environ['HOME'], 'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min')
IMAGELIST = osp.join(os.environ['HOME'], 'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt')
INPUT_NODES = 'input'

CMD_SEQ = {
    'resnet50_v1_tf': {
        'means': [103.94, 116.78, 123.68], # BGR
        'scales': [1.0, 1.0, 1.0],
        'channel_swap': True,
        'resize_crop': True,
        'center_crop': 1,
        'width': 224,
        'height': 224,
    },
    'inception_v1_tf': {
        'means': [127.5, 127.5, 127.5], # BGR
        'scales': [2.0/255, 2.0/255, 2.0/255],
        'channel_swap': True,
        'resize_crop': False,
        'center_crop': 0.875,
        'width': 224,
        'height': 224,
    },
    'inception_v3_tf': {
        'means': [127.5, 127.5, 127.5], # BGR
        'scales': [2.0/255, 2.0/255, 2.0/255],
        'channel_swap': True,
        'resize_crop': False,
        'center_crop': 0.875,
        'width': 299,
        'height': 299,
    },
}

def center_crop_img(image, factor):
    assert factor <= 1
    h, w, _ = image.shape
    hh = int(h * factor)
    ww = int(w * factor)
    offset_h = (h - hh) // 2
    offset_w = (w - ww) // 2
    return image[offset_h:offset_h+hh, offset_w:offset_w+ww,:]

def resize_img(image, resize_crop, width, height):
    if not resize_crop:
        image = cv2.resize(image, (width, height))
    else:
        h, w, _ = image.shape
        scale_h = height / float(h)
        scale_w = width / float(w)
        scale = max(scale_h, scale_w)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        offset_h = (image.shape[0] - height) // 2
        offset_w = (image.shape[1] - width) // 2
        image = image[offset_h:offset_h+height, offset_w:offset_w+width, :]
    return image

def preprocess_one_image_fn(image_path, fn):
    width = fn['width']
    height = fn['height']

    image = cv2.imread(image_path).astype('float32')
    image = center_crop_img(image, fn.get('center_crop', 1))
    image = resize_img(image, fn.get('resize_crop', False), width, height)
    image = (image - fn['means']) * fn['scales']
    if fn.get('channel_swap', False):
        return image[:,:,::-1]
    else:
        return image

def load_image(iter, image_list = None, pre_process_function = None, input_nodes = 'data', batch_size = 1):
    images = []
    labels = []
    for i in range(batch_size):
        image_name, label  = image_list[iter * batch_size + i].split(' ')
        print((image_name))
        image = preprocess_one_image_fn(image_name, pre_process_function)
        images.append(image)
    return {input_nodes: images}

def get_input_fn(pre_processing_function_name, imageroot = IMAGEROOT, imagelist = IMAGELIST, input_nodes = INPUT_NODES):
    cmd_seq = CMD_SEQ[pre_processing_function_name]
    with open(imagelist) as fin:
        lines = fin.readlines()
        image_list = list(map(lambda x:os.path.join(imageroot, x.strip()), lines))
    return partial(load_image, image_list = image_list, pre_process_function = cmd_seq, input_nodes = input_nodes)

for name in CMD_SEQ:
    globals()['input_fn_{}'.format(name)] = get_input_fn(name)
