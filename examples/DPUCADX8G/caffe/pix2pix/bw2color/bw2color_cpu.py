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


## pix2pix caffe interference 
# b/w to color 

#%% import package

import argparse
import caffe
import cv2
import numpy as np
import os


import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import skimage.io as io

#%% define functions

def norm_image(IMG):
    # output scale: [0,1]
    output = (IMG - np.min(IMG))/(np.max(IMG)-np.min(IMG)) 
    # normalize [0,255]
    output1 = output*255
    # assure integer 8bit
    output1 = output1.astype('uint8')    
    return output1



#%% main 
if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default="./test_output/", help='Optionally, save all generated outputs in specified folder')
    parser.add_argument('--image', required=True,
                        help='User can provide an image to run')
    args = vars(parser.parse_args())

    VAI_ALVEO_ROOT = os.environ["VAI_ALVEO_ROOT"]
    os.makedirs(args['output_path'], exist_ok=True) 

    # model configuration
    model_def = './quantize_results/deploy.prototxt'
    model_weights = './quantize_results/deploy.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST) 
    input_shape = (256, 256)

    image_path = args["image"]
    # load image
    image = plt.imread(image_path)
    if image.shape[:2] != input_shape:
        print("[INFO] Resizing the input image to {}".format(input_shape))
        image = cv2.resize(image, input_shape)
    ## preprocessing
    # add one dimension
    batch_A = np.expand_dims(image,0)
    # normalize [0,255] --> [-1,1]
    batch_A1 = (batch_A / 127.5) - 1  
    # channel transpose NHWC to NCHW
    batch_A2 = np.transpose(batch_A1,(0,3,1,2))

    ## net forward (feed into caffe network)
    net.blobs['input_3'].data[...] = batch_A2
    net.forward()
    fake_B = net.blobs['activation_10'].data

    ## post processing
    # normalize output [0,255]
    fake_B1 = norm_image(np.transpose(fake_B[0,:,:,:],(1,2,0)))
    # save the output image as file
    image_name = os.path.basename(image_path)
    name, ext = os.path.splitext(image_name)
    out_name = name + '_cpu.jpg'
    out_path = os.path.join(args['output_path'], out_name)
    io.imsave(out_path, fake_B1)       
    print('[INFO] output file is saved in ' + args["output_path"])
