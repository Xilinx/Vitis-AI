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
Compile TVM model for Xilinx Vitis-AI acceleration
==================================================

This example shows how to build a TVM convolutional neural network 
model with Relay for Vitis-AI acceleration

Setup: 
    - Add imagenet validation subset for calibration in imagenet/val-small

"""

import os
import numpy as np
import cv2
import time
from typing import List
from pathlib import Path

import pyxir
import pyxir.contrib.target.DPUCADX8G
import pyxir.contrib.target.DPUCZDX8G

import logging

import tvm
from tvm import contrib
import tvm.relay as relay
from tvm.relay import transform
from tvm.contrib import utils, graph_runtime
from tvm.contrib.target import vitis_ai
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.vitis_ai import annotation

FILE_DIR   = os.path.dirname(os.path.abspath(__file__))
HOME_DIR   = os.getenv('HOME')
QUANT_DIR = os.path.join(str(Path.home()),
                        'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/')


if not os.path.exists(QUANT_DIR):
    raise ValueError("Could not find directory "
                     "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/."
                     " Please install using following commands before"
                     " running this example: \n"
                     " $ python3 -m ck pull repo:ck-env\n"
                     " $ python3 -m ck install package:imagenet-2012-val-min")
    

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
###############################################################################
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image

#from matplotlib import pyplot as plt
block = get_model('resnet18_v1', pretrained=True)
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

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

###############################################################################
# MODEL SETTINGS
#
# Parameter settings for compiling a model using tvm-vai flow
# quant_dir      : path to images for quantization
# target         : hardware accelerator to run the compiled model
#                      options: 'DPUCADX8G', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102'
# tvm_target     :
# lib_kwargs     : 

###############################################################################

input_name  = 'data'
input_shape = (1,3,224,224)
shape_dict  = {input_name:input_shape}
target      = 'DPUCADX8G'
tvm_target  = 'llvm'
lib_kwargs  = {}

###############################################################################
# INPUTS FUNC
#
# Define and inputs function which takes in an iterator value and returns a
# dictionary mapping from input name to array containing dataset inputs. Note 
# that the input function should always return image data in NCHW layout as 
# all models are converted to NCHW layout internally for Vitis-AI compilation.
# 
# This is necessary for quantizating the model for acceleration using Vitis-AI.
###############################################################################

def inputs_func(img_files: List[str]):
    inputs = []
    for img_path in img_files:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(input_shape[2:])
       
        inputs.append(transform_image(img))
    return inputs

###############################################################################
# PARTITION & BUILD
# 
# Use TVM Module pass to annotate and partition Relay graph for Vitis-AI acceleration. Targets can be 'DPUCADX8G', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102'
# Afterwards build graph using standard TVM flow.
##############################################################################



mod, params = relay.frontend.from_mxnet(block, shape_dict)
mod["main"] = bind_params_by_name(mod["main"], params)
mod = annotation(mod, params, target)
mod = relay.transform.MergeCompilerRegions()(mod)
mod = relay.transform.PartitionGraph()(mod)

vai_build_dir = os.path.join(os.getcwd(), target + '_build')
vai_work_dir = os.path.join(os.getcwd(), target + '_work')
export_rt_mod_file = os.path.join(os.getcwd(), 'vitis_ai.rtmod')
with tvm.transform.PassContext(opt_level=3,
                               config={'relay.ext.vitis_ai.options.target': target,
                                       'relay.ext.vitis_ai.options.build_dir': vai_build_dir,
                                       'relay.ext.vitis_ai.options.work_dir': vai_work_dir,
                                       'relay.ext.vitis_ai.options.export_runtime_module': export_rt_mod_file}):   
	lib = relay.build(mod, tvm_target, params=params)



############################################################
## Quantization using first N inputs
## 
## Usually, to be able to accelerate inference of Neural 
## Network models with Vitis-AI DPU accelerators, those models 
## need to quantized upfront. In the ONNXRuntime Vitis-AI 
## execution provider we make use of On-The-Fly (OTF) Quantization 
## to remove this additional preprocessing step. In this flow,
## one doesn't need to quantize his/her model upfront but can 
## make use of the typical inference execution calls 
## (InferenceSession.run) to quantize the model on-the-fly 
## using the first N inputs. This will set up and calibrate
## the Vitis-AI DPU and from that point onwards inference 
## will be accelerated for all next inputs.
## Set the number of inputs used for quantization to e.g. 8 
## using the PX_QUANT_SIZE environment variable if you want
## to quantize on fewer inputs. The default is 128.
############################################################

print("Create InferenceSession for OTF Quantization")
InferenceSession = graph_runtime.GraphModule(lib["default"](tvm.cpu()))

px_quant_size = int(os.environ['PX_QUANT_SIZE']) \
    if 'PX_QUANT_SIZE' in os.environ else 128

print("Start OTF Quantization on first {} images".format(px_quant_size))

quant_files = [os.path.join(QUANT_DIR, f) for f in os.listdir(QUANT_DIR)
             if f.endswith(('JPEG', 'jpg', 'png'))][:px_quant_size]
quant_images = inputs_func(quant_files)
print('Loaded {} inputs successfully.'.format(len(quant_images)))

for i in range(px_quant_size):
    InferenceSession.set_input(input_name, quant_images[i]) 
    # print("running") 
    InferenceSession.run()

print("Finished OTF Qunatization")

#########################################################
# Export compiled model for execution #
#########################################################

if target.startswith('dpuv2') or target.startswith('DPUCZDX8G'):
    # Export runtime module
    temp = utils.tempdir()
    lib.export_library(temp.relpath("tvm_lib.so"))

    # Build and export lib for aarch64 target
    tvm_target = tvm.target.arm_cpu('ultra96')
    lib_kwargs = {
        'fcompile': contrib.cc.create_shared,
        'cc': "/usr/aarch64-linux-gnu/bin/ld"
    }

    with tvm.transform.PassContext(opt_level=3,
                                   config={'relay.ext.vitis_ai.options.load_runtime_module': export_rt_mod_file}):
        lib_dpuv2 = relay.build(mod, tvm_target, params=params)

    lib_dpuv2.export_library('tvm_dpu_cpu.so', **lib_kwargs)

else:
    lib.export_library('tvm_dpu_cpu.so')

