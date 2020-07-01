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


import random
from PIL import Image
import math
import torch
import numpy as np
import numbers
from torchvision.transforms import Pad
from torchvision.transforms import functional as F

MEAN = [.485, .456, .406]
STD =  [.229, .224, .225]

class Resize(object):
    '''
        Resize the images
    '''
    def __init__(self, size=(512, 512)):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, rgb_img, label_img):
        rgb_img = rgb_img.resize(self.size, Image.BILINEAR)
        label_img = label_img.resize(self.size, Image.NEAREST)
        return rgb_img, label_img


class Resize_eval(object):
    '''
        Resize the images
    '''
    def __init__(self, size=(512, 512)):

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, rgb_img, label_img):
        #only resize image
        rgb_img = rgb_img.resize(self.size, Image.BILINEAR)
        return rgb_img, label_img

class RandomFlip(object):
    '''
        Random Flipping
    '''
    def __call__(self, rgb_img, label_img):
        if random.random() < 0.5:
            rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
            label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
        return rgb_img, label_img

class Normalize(object):
    '''
        Normalize the tensors
    '''
    def __call__(self, rgb_img, label_img=None):
        rgb_img = F.to_tensor(rgb_img) # convert to tensor (values between 0 and 1)
        rgb_img = F.normalize(rgb_img, MEAN, STD) # normalize the tensor
        label_img = torch.LongTensor(np.array(label_img).astype(np.int64))
        return rgb_img, label_img


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
