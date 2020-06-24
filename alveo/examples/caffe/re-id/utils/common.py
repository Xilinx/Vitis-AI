# (c) Copyright 2019 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.

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
