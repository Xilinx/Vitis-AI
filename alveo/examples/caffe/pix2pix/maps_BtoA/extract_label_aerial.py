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

# extract label and photo from facades images

#%% import package

#import numpy as np
import cv2
import os 

#import caffe
#import matplotlib.pyplot as plt
import skimage.io as io

#%% define functions

def load_images(PATH,fn):
    # load image
    img = cv2.imread(PATH+fn)
    

    # split for BtoA
    img_A = img[:,:600,:] 
    img_B = img[:,600:,:]  

           
    # BGR to RGB
    img_A1 = img_A[...,::-1]    
    img_B1 = img_B[...,::-1]  

    return img_A1, img_B1



#%% main 
    

# file path facades
Image_path = './maps/val/'
output_pathA = Image_path+'photo/'
output_pathB = Image_path+'label/'

if not os.path.exists(output_pathA):
    os.mkdir(output_pathA)
    
if not os.path.exists(output_pathB):
    os.mkdir(output_pathB)
    
# load image
for i in range(1098):
    fn = str(i+1)+'.jpg'    
    imageA, imageB = load_images(Image_path,fn)
    io.imsave(output_pathA+str(i+1)+'.jpg',imageA)
    io.imsave(output_pathB+str(i+1)+'.jpg',imageB)
 