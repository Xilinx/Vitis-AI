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
import math
import numpy as np
from tqdm import tqdm


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h, w, 3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image


def resize_image(image, model_image_size):
    if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
        boxed_image = letterbox_image(image, new_image_size)

    boxed_image = np.array(boxed_image, dtype='float32')
    boxed_image /= 255.

    return boxed_image


def calib_input(eval_image_path, eval_image_list,
                input_height, input_width, iter_num=None):
    with open(eval_image_list) as fr:
        lines = fr.readlines()
    img_list = []
    img_ids = []
    img_input_shapes = []
    load_size = len(lines) if iter_num is None else iter_num
    for i in tqdm(range(0, load_size)):
        line = lines[i]
        img_id = line.strip()
        img_path = os.path.join(eval_image_path, img_id + '.jpg')
        image = cv2.imread(img_path)
        # BGR -> RGB
        image = image[..., ::-1]
        resize_img = resize_image(image, (input_height, input_width))
        img_list.append(resize_img)
        img_ids.append(img_id)
        # ignore the channel dimension
        img_input_shapes.append(image.shape[0:-1])
    return (img_list, img_ids, img_input_shapes)
