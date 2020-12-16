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

from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def segmentation_collate(batch):
    imgs = []
    segs = []
    for sample in batch:
        imgs.append(sample[0])
        segs.append(sample[1])
    return torch.stack(imgs, 0), torch.stack(segs, 0)

def oridetection_collate(batch):
    targets = []
    imgs = []
    oris = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        oris.append(torch.FloatTensor(sample[2]))
    return torch.stack(imgs, 0), targets, oris

def base_transform(image, size, mean):
    x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        return base_transform(image, self.size, self.mean)

