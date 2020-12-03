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

# convert rgb to b/w

#%% import package
import argparse
#import numpy as np

from skimage import color
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import skimage.io as io

#%% main 
if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True,
                        help='User can provide an image to run')
    args = vars(parser.parse_args())
    
    
    image_path = args["image"]
    # load image
    image = plt.imread(image_path)  # [0,255]
    # convert rgb to gray [0,1]
    img_gray1 = color.rgb2gray(image)    
    # convert 1channel to 3 channel [0,1]
    img_gray2 = color.gray2rgb(img_gray1)
    img_gray3 = img_gray2*255
    img_gray3 = img_gray3.astype('uint8')
    fn = image_path.split('.')[0]
    io.imsave(fn+'_bw.jpg',img_gray3)


    
    
