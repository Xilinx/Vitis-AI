#!/usr/bin/env python
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
import numpy as np

def normalize(x):
    return  x / np.linalg.norm(x)
    
def cosine_distance(feat1, feat2):
    return 1 - np.dot(feat1, feat2.transpose())/(np.linalg.norm(feat1) * np.linalg.norm(feat2) )

def process_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.resize(im, (80,160))
    im = im[...,::-1]
    im = np.transpose(im, [2,0,1])
    scale = 0.017429
    im = im*1.0
    im[0] = (im[0]-123.0)*scale
    im[1] = (im[1]-116.0)*scale 
    im[2] = (im[2]-103.0)*scale
    im = np.expand_dims(im, axis=0)
    return im

def get_batch_images(image_list):
    imgs = process_image(image_list[0])
    for img_path in image_list[1:]:
        img = process_image(img_path)
        imgs = np.concatenate((imgs, img), axis=0) 
    return imgs

def bn(x, mean, var, weight ):
    return (x-mean)/np.sqrt(var + 1e-5) * weight
