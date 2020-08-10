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



#%% import packages

import os
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import skimage.io as io


#%% check mult-class segmentation images
dir_data = "./PhC-C2DH-U373/"        
dir_seg = dir_data + "Seg/"
dir_img = dir_data + "Img/"

img_dir = os.path.join(dir_data, '01/')
seg_dir = os.path.join(dir_data, '01_ST/SEG/')

if not os.path.exists(dir_seg):
    os.mkdir(dir_seg)
    
if not os.path.exists(dir_img):
    os.mkdir(dir_img)
    
for filename in os.listdir(img_dir):
    fn = filename.split('.')[0]
    img = plt.imread(img_dir+filename)
    io.imsave(dir_img+fn+'.png', img)

        
for filename in os.listdir(seg_dir):
    fn = filename.split('.')[0].split('seg')[1]
    seg = plt.imread(seg_dir+filename)
    io.imsave(dir_seg+'t'+fn+'.tif', seg)        

