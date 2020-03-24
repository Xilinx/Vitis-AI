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


## semantic segmenation


#%% import packages
import numpy as np
from PIL import Image
import argparse
import os
#import sys
#import glob
import cv2
#import time
import matplotlib.pyplot as plt
import caffe


#%% define functions


def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])
    return img_color

def segment(net, img_file):
    IMG_MEAN = np.array((104,117,123))  # mean_values for B, G,R
    INPUT_W, INPUT_H = 512, 256 # W, H  512, 256
    TARGET_W, TARGET_H = 2048, 1024
    
    im_ = cv2.imread(img_file)
    w, h = TARGET_W, TARGET_H
    in_ = cv2.resize(im_, (INPUT_W, INPUT_H))
    in_ = in_ * 1.0
    in_ -= IMG_MEAN
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    

    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)

    #save color_output
    pred_label_color = label_img_to_color(out)
    color_to_save = Image.fromarray(pred_label_color.astype(np.uint8))
    color_to_save = color_to_save.resize((w,h))
    return color_to_save

    
#%% main 
if __name__ == "__main__":    
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default="./test_output/", help='Optionally, save all generated outputs in specified folder')
    parser.add_argument('--image', default=None, help='User can provide an image to run')
    args = vars(parser.parse_args())
    
    VAI_ALVEO_ROOT=os.environ["VAI_ALVEO_ROOT"]
    if not os.path.isdir(args["output_path"]):
        os.mkdir(args["output_path"])

    # model configuration
    model_def = 'xfdnn_deploy.prototxt'
    model_weights = VAI_ALVEO_ROOT+'/examples/caffe/models/FPN_CityScapes/deploy.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST) 

    
    if args["image"]:
        fn = args["image"]
        # create segmentation image
        semantic_seg = segment(net, fn)
        # save output
        filename = 'seg_'+fn
        semantic_seg.save(args["output_path"]+filename)
        print('output file is saved in '+args["output_path"])
    else:
        print('Please provide input image as "--image filename"' )



