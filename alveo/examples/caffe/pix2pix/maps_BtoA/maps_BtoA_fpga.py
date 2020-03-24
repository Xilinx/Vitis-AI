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
# maps B to A

#%% import package

import numpy as np
import cv2
import os
import caffe
#import matplotlib.pyplot as plt
import skimage.io as io
import argparse


#%% define functions



def load_images(fn):
    # load image
    img = cv2.imread(fn)
    
    # resize as 256 x 256
    img_A256 = cv2.resize(img,(256,256) )
      
    # BGR to RGB
    img_A1 = img_A256[...,::-1]    
               
    # normalize [-1,1]
    img_A2 = (img_A1 / 127.5) - 1
    
    # channel transpose NHWC to NCHW
    img_A3 = np.transpose(img_A2,(2,0,1))
    
    return img_A3


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
    parser.add_argument('--image', default=None, help='User can provide an image to run')
    args = vars(parser.parse_args())
    
    VAI_ALVEO_ROOT=os.environ["VAI_ALVEO_ROOT"]
    if not os.path.isdir(args["output_path"]):
        os.mkdir(args["output_path"])  
        
    # model configuration
    model_def = 'xfdnn_deploy.prototxt'
    model_weights = VAI_ALVEO_ROOT+'/examples/caffe/models/maps_BtoA/deploy.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST) 

    
    if args["image"]:
        fn = args["image"]
        # load image
        image = load_images(fn)

        ## preprocessing
        # add one dimension
        batch_A = np.expand_dims(image,0)

        ## net forward (feed into caffe network)
        net.blobs['input_3'].data[...] = batch_A
        net.forward()
        fake_B = net.blobs['activation_10'].data

        ## post processing
        # normalize output [0,255]
        fake_B1 = norm_image(np.transpose(fake_B[0,:,:,:],(1,2,0)))
        # save the output image as file
        filename = 'output_'+fn
        io.imsave(args["output_path"]+filename,fake_B1)       
        print('output file is saved in '+args["output_path"])
    else:
        print('Please provide input image as "--image filename"' )
