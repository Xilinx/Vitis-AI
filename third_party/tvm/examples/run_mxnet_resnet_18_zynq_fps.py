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
Run TVM model with Xilinx Vitis AI acceleration
===============================================

This example shows how to run MxNet ResNet 18 model
built with TVM for Vitis AI acceleration
"""
import os
import time
import argparse
import numpy as np
import threading
import multiprocessing as mp

import pyxir
import tvm
from tvm.contrib import graph_executor

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


###########################################################
# Main and run functions
###########################################################

def run(mod, nb_images, inputs):
    for _ in range(nb_images):
        for name, data in inputs.items():
            mod.set_input(name, data)
        mod.run()


def main(lib_path, shape_dict, threadnum, iterations, nb_tvm_threads=None):
    # Load image and do preprocessing
    img_shape = shape_dict[list(shape_dict.keys())[0]][2:]
    image = Image.open(img_path).resize(img_shape)
    image = transform_image(image)
    inputs = {}
    inputs[list(shape_dict.keys())[0]] = image
    
    # Tune number of TVM CPU threads being used to get best performance.
    # Setting this to 0 uses all available threads.
    config_threadpool = tvm.get_global_func("runtime.config_threadpool")
    if nb_tvm_threads is not None:
        config_threadpool(1, nb_tvm_threads)

    # Create TVM runtime modules
    mods = []
    for i in range(threadnum):
        lib = tvm.runtime.module.load_module(lib_path)
        mod = graph_executor.GraphModule(lib["default"](tvm.cpu()))
        mods.append(mod)
    
    threads = []
    time_start = time.time()
    for i in range(threadnum):
        t = mp.Process(target=run, args=(mods[i], iterations, inputs))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    time_end = time.time()
    total = iterations * threadnum
    time_total = time_end - time_start
    print("Total", total, "Time", time_total)
    fps = float(total / time_total)
    print("%.2f FPS" % fps)

    del mods
        

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
# -f           : Path to the exported TVM compiled model (tvm_dpu_cpu.so in the example)
# -t           : Number of threads to use
# --iterations : The number of iterations to run the model
# --nb_tvm_threads : The number of CPU threads being used by TVM. You might have to limit this number
#                      to get best performance.
#
# example:
# ./run_mxnet_resnet_18_fps.py -f /PATH_TO_DIR -t 500 --iterations 1

##############################################################################
 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Path to TVM library file (.so)", default=FILE_DIR)
    parser.add_argument("-t", help="Number of threads", default=1, type=int)
    parser.add_argument("--iterations", help="The number of iterations to run.", default=1000, type=int)
    parser.add_argument("--nb_tvm_threads", help="The number of CPU threads being used by TVM.", default=None, type=int)
    args = parser.parse_args()
    lib_path = args.f if os.path.isabs(args.f) else os.path.join(os.getcwd(), args.f)
    threadnum = args.t
    iterations = args.iterations
    nb_tvm_threads = args.nb_tvm_threads
    shape_dict = {'data': [1, 3, 224, 224]}
    main(lib_path, shape_dict, threadnum, iterations, nb_tvm_threads)
