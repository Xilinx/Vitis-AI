# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Run TVM model for Xilinx Vitis-AI acceleration
==================================================

This example shows how to run MxNet Resent_18 model
 built with TVM for Vitis-AI acceleration

"""

import os
import argparse
import numpy as np
import time


import pyxir
import tvm
from tvm.contrib import graph_runtime
from tvm.contrib.target import vitis_ai

from PIL import Image
from tvm.contrib.download import download_testdata

FILE_DIR = os.path.dirname(os.path.abspath(__file__))



######################################################################
# Download Test Image
######################################################################

img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_name = 'cat.png'
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'imagenet1000_clsid_to_human.txt'
img_path = download_testdata(img_url, 'cat.png', module='data')
synset_path = download_testdata(synset_url, synset_name, module='data')
with open(synset_path) as f:
    synset = eval(f.read())

###########################################################
# Define utility functions
###########################################################

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def softmax(x):        
        x_exp = np.exp(x - np.max(x))
        return x_exp / x_exp.sum()

def run(fdir,shape_dict, iterations):

    
    # DOWNLOAD IMAGE FOR TEST
    img_shape=shape_dict[list(shape_dict.keys())[0]][2:]
    print(img_shape)
    image = Image.open(img_path).resize(img_shape)
    
    # IMAGE PRE-PROCESSING
    image = transform_image(image)
    
    # RUN #
    inputs = {}
    inputs[list(shape_dict.keys())[0]] = image

    # load the pre-compiled module into memory
    lib = tvm.runtime.load_module(os.path.join(fdir,"tvm_dpu_cpu.so"))
    module = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
    module.set_input(**inputs)
  


    # VAI FLOW
    for i in range(iterations):
        start = time.time()
        module.run()
        stop = time.time()
        res = [module.get_output(idx).asnumpy()
                for idx in range(module.get_num_outputs())]
        
        inference_time = np.round((stop - start) * 1000, 2)
        
        res = softmax(res[0])
        top1 = np.argmax(res)
        print('========================================')
        print('TVM prediction top-1:', top1, synset[top1])
        print('========================================')
        
        print('========================================')
        print('Inference time: ' + str(inference_time) + " ms")
        print('========================================')
        

###############################################################################
# RUN MXNET_RESNET_18
# 
# Before running the mxnet_resnet_18 model, you have to compile the model
# using the examples/compile_mxnet_resnet_18.py script
#
# The compile script generates an output file name "tvm_dpu_cpu.so"
# Once you setup your device, you could copy the compiled file to your acceleration device and run the model as follows:
#
# Parameter settings for the run script:
# -f           : Path to directory containing TVM compiled model
# --iterations : The number of iterations to run the model
#
# example:
# ./run_mxnet_resnet_18.py -f /PATH_TO_DIR --iterations 1

##############################################################################
 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Path to directory containing TVM compilation files", default=FILE_DIR)
    parser.add_argument("--iterations", help="The number of iterations to run.", default=2, type=int)
    args = parser.parse_args()
    fdir = args.f if os.path.isabs(args.f) else os.path.join(os.getcwd(), args.f)
    iterations = args.iterations
    shape_dict = {'data': [1, 3, 224, 224]}
    
    run(fdir, shape_dict, iterations)
