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

from __future__ import division
import torch
import torchvision
import cv2
import math
import random
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from torchvision.transforms import functional as F

__all__ = ["Gamma", "RandomGamma", "RandomChoiceGamma", "GammaBalance"]

class Gamma(object):
    def __init__(self, value):
        """
        Performs Gamma Correction on the input image. Also known as 
        Power Law Transform. This function transforms the input image 
        pixelwise according 
        to the equation Out = In**gamma after scaling each 
        pixel to the range 0 to 1.

        Arguments
        ---------
        value : float
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.value = value

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be adjusted.

        Returns:
            PIL Image: Adjusted image.
        """
        return F.adjust_gamma(img, self.value)

class RandomGamma(object):

    def __init__(self, min_val, max_val):
        """
        Performs Gamma Correction on the input image with some
        randomly selected gamma value between min_val and max_val. 
        Also known as Power Law Transform. This function transforms 
        the input image pixelwise according to the equation 
        Out = In**gamma after scaling each pixel to the range 0 to 1.

        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range

        NOTE:
        for values:
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.values = (min_val, max_val)

    def __call__(self, img):
        value = random.uniform(self.values[0], self.values[1])
        return Gamma(value)(img)

class RandomChoiceGamma(object):

    def __init__(self, values, p=None):
        """
        Performs Gamma Correction on the input image with some
        gamma value selected in the list of given values.
        Also known as Power Law Transform. This function transforms 
        the input image pixelwise according to the equation 
        Out = In**gamma after scaling each pixel to the range 0 to 1.

        Arguments
        ---------
        values : list of floats
            gamma values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.

        NOTE:
        for values:
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.values = values
        self.p = p

    def __call__(self, img):
        value = th_random_choice(self.values, p=self.p)
        return Gamma(value)(img)

class GammaBalance(object):

    def __init__(self):
        """
        Performs Gamma Correction To Balance Gamma 
        """
        pass

    def __call__(self, img):
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        bright_thres = 0.5
        dark_thres = 0.5
        total_pixel = np.size(gray)
        dark_pixel = np.sum(hist[:56])
        bright_pixel = np.sum(hist[200:256])
        bright_ratio = bright_pixel / total_pixel
        dark_ratio = dark_pixel / total_pixel
        if dark_ratio > dark_thres: # dark
            img_g = Gamma(max(0.7-dark_ratio+dark_thres, 0.01))(img) 
        elif bright_ratio > bright_thres: # bright
            img_g = Gamma(1.3+bright_ratio-bright_thres)(img)
        else:
            img_g = img.copy()
        return img_g


