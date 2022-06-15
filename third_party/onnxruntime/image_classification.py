# Copyright 2020 Xilinx Inc.
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

"""
ONNX Runtime with Xilinx Vitis AI acceleration example
======================================================

Microsoft ONNX Runtime is a framework designed for running ONNX models
on a variety of platforms.

ONNX Runtime is enabled with Vitis AI and available through the Microsoft
github page: https://github.com/microsoft/onnxruntime

Follow the setup instructions below before running this example.
"""


###########################################################
# SETUP
#
# 1. Download minimal ImageNet validation dataset (step specific to this example):
#    ```
#    python3 -m ck pull repo:ck-env
#    python3 -m ck install package:imagenet-2012-val-min
#    ```
# 2. (Optional) set the number of inputs to be used for on-the-fly quantization to a lower number (e.g. 8) to decrease the quantization time (potentially at the cost of lower accuracy):
#    ```
#    export PX_QUANT_SIZE=8
#    ```
# 3. Run the ResNet 18 example script:
#    ```
#    cd /workspace/external/onnxruntime
#    python3 image_classification.py [DPU TARGET ID]
#    ```
#    Where the DPU target identifier can be found in the [DPU Targets](#dpu-targets) table above.
#    After the model has been quantized and compiled using the first N inputs you should see accelerated execution of the 'images/dog.jpg' image with the DPU accelerator.
###########################################################

import os
import cv2
import sys
import json
import onnx
import time
import urllib
import numpy as np

import pyxir
import pyxir.frontend.onnx
import pyxir.contrib.dpuv1.dpuv1
import onnxruntime

from typing import List
from PIL import Image
from pathlib import Path


DATA_DIR = os.path.join(str(Path.home()),
                        'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/')

if not os.path.exists(DATA_DIR):
    raise ValueError("Could not find directory "
                     "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/."
                     " Please install using following commands before"
                     " running this example: \n"
                     " $ python3 -m ck pull repo:ck-env\n"
                     " $ python3 -m ck install package:imagenet-2012-val-min")
if not os.path.exists('images/dog.jpg'):
    raise ValueError("This example requires a dog.jpg image in the ./images"
                     " directory")


###########################################################
# Download ResNet 18 model from ONNX model zoo
###########################################################

onnx_model_url = "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.tar.gz"
imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

if not os.path.isfile('resnet18v1/resnet18v1.onnx'):
    urllib.request.urlretrieve(onnx_model_url, filename="resnet18v1.tar.gz")
    os.system(u'tar -xvzf resnet18v1.tar.gz --warning=no-unknown-keyword')
if not os.path.isfile('imagenet-simple-labels.json'):
    urllib.request.urlretrieve(imagenet_labels_url,
                               filename="imagenet-simple-labels.json")

model_path = 'resnet18v1/resnet18v1.onnx'


###########################################################
# Define utility functions
###########################################################

def load_labels(path: str):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def softmax(x: np.ndarray):
    x_exp = np.exp(x - np.max(x))
    return x_exp / x_exp.sum()

def resize_smallest_side(img: np.ndarray, size: int):

    def get_size(height: int, width: int, aspect_ratio: float):
        return (int(round(width * aspect_ratio)), int(round(height * aspect_ratio)))

    smallest_side_size = img.shape[0] if img.shape[0] < img.shape[1] \
        else img.shape[1]
    aspect_ratio = float(size) / float(smallest_side_size)

    new_size = get_size(img.shape[0], img.shape[1], aspect_ratio)
    return cv2.resize(img, new_size, cv2.INTER_LINEAR)

def central_crop(img: np.ndarray, size: List[int]):
    # !! img should be in HWC layout
    img_h, img_w, _ = img.shape
    height, width = size

    if height > img_h:
        raise ValueError("Provided crop height is larger than provided"
                         " image height.")
    if width > img_w:
        raise ValueError("Provided crop width is larger than provided"
                         " image width.")

    start_h, start_w = int((img_h - height) / 2), int((img_w - width) / 2)
    end_h, end_w = start_h + height, start_w + width

    return img[start_h:end_h, start_w:end_w, :]

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def postprocess(result: np.ndarray):
    return softmax(np.array(result)).tolist()

def inputs_func(img_files: List[str]):
    inputs = []
    for img_path in img_files:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_data = resize_smallest_side(np.array(img, dtype=np.float32), 224)
        img_data = central_crop(img_data, (224, 224))
        img_data = np.array(img_data).transpose(2, 0, 1)
        inputs.append(preprocess(img_data))
    return inputs


###########################################################
# Create InferenceSession
###########################################################

target = str(sys.argv[1])

print("Create InferenceSession")
vitis_ai_provider_options = {'target': target, 'export_runtime_module': 'vitis_ai.rtmod'}
session = onnxruntime.InferenceSession(model_path, None, ["VitisAIExecutionProvider"],
                                       [vitis_ai_provider_options])

# get the name of the first input of the model
input_name = session.get_inputs()[0].name


###########################################################
# Quantization using first N inputs
# 
# Usually, to be able to accelerate inference of Neural 
# Network models with Vitis-AI DPU accelerators, those models 
# need to quantized upfront. In the ONNXRuntime Vitis-AI 
# execution provider we make use of on-the-fly quantization 
# to remove this additional preprocessing step. In this flow,
# one doesn't need to quantize his/her model upfront but can 
# make use of the typical inference execution calls 
# (InferenceSession.run) to quantize the model on-the-fly 
# using the first N inputs. This will set up and calibrate
# the Vitis-AI DPU and from that point onwards inference 
# will be accelerated for all next inputs.
###########################################################

# Set the number of inputs used for quantization to e.g. 8 
# using the PX_QUANT_SIZE environment variable if you want
# to quantize on fewer inputs. The default is 128.
px_quant_size = int(os.environ['PX_QUANT_SIZE']) \
    if 'PX_QUANT_SIZE' in os.environ else 128

print("Quantize on first {} inputs".format(px_quant_size))

file_dir = DATA_DIR
img_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir)
             if f.endswith(('JPEG', 'jpg', 'png'))][:px_quant_size]
inputs = inputs_func(img_files)
print('Loaded {} inputs successfully.'.format(len(inputs)))

outputs = [session.run([], {input_name: inputs[i]})[0] 
                            for i in range(px_quant_size)]


###########################################################
# Accelerated inference on new image
###########################################################

input_data = inputs_func(['images/dog.jpg'])[0]

start = time.time()
raw_result = session.run([], {input_name: input_data})
end = time.time()


###########################################################
# Postprocessing
###########################################################

labels = load_labels('imagenet-simple-labels.json')

res = postprocess(raw_result)


inference_time = np.round((end - start) * 1000, 2)
idx = np.argmax(res)

print('========================================')
print('Final top prediction is: ' + labels[idx])
print('========================================')

print('========================================')
print('Inference time: ' + str(inference_time) + " ms")
print('========================================')

sort_idx = np.flip(np.squeeze(np.argsort(res)))
print('============ Top 5 labels are: ============================')
print(labels[sort_idx[:5]])
print('===========================================================')
