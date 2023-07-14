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

This script is supplementry to relay_yolov3 notebook that demonstrate how to run the
compiled custom tensoflow YoloV3 model that was compiled using the TVM - Vitis AI
acceleration flow on a Zynq edge device.
"""

import os, sys
import argparse
import numpy as np
import time

import cv2
import pyxir
import tvm
from tvm.contrib import graph_executor

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists('/home/root/tensorflow-yolov3/'):
    raise ValueError("'tensorflow-yolov3' repo is required for running this example\n"\
                     "Please run the setup script: `bash setup_custom_yolov3_zynq.sh`")
    
###########################################################
# Define utility functions
###########################################################

def transform_image(image):
    """Data preprocessing function"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = (320, 320)
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.

    return image_padded


###########################################################
# Main function
###########################################################

def run(file_path, shape_dict, iterations):
    # IMAGE PRE-PROCESSING
    img_path = "/home/root/tensorflow-yolov3/docs/images/road.jpeg"
    original_image = cv2.imread(img_path)
    image = transform_image(original_image) 
    image = np.array(image)[np.newaxis, :]

    inputs = {}
    inputs[list(shape_dict.keys())[0]] = image

    # load the pre-compiled module into memory
    print ("Loading the compiled model ...")
    lib = tvm.runtime.load_module(file_path)
    module = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    module.set_input(**inputs)

    if '/home/root/tensorflow-yolov3' not in sys.path:
        sys.path.append('/home/root/tensorflow-yolov3/')
    import core.utils as utils
    num_classes = 80
    input_size  = 320

    print ("Running the compiled model ...")
    for i in range(iterations):
        # RUN 
        start = time.time()
        module.run()
        stop = time.time()
        inference_time = np.round((stop - start) * 1000, 2)

        # POST PROCESSING
        res  = module.get_output(0).asnumpy()
        pred_bbox   = res.reshape(-1,85)
        original_image_size = original_image.shape[:2]
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(original_image, bboxes)
        cv2.imwrite("./output.png", (image))
        print('========================================')
        print('Detection output stored in ./output.png')
        print('========================================')
        print('Inference time: ' + str(inference_time) + " ms")
        print('========================================')
       

###############################################################################
# RUN CUSTOM YOLOV3
# 
# Before running the custom YoloV3 model, you have to compile the model
# using the notebook in examples/external_yolov3_tutorial.ipynb notebook
#
# If the model is compiled for an edge device (DPUCZDX8G-zcu104, DPUCZDX8G-zcu102,
#   DPUCZDX8G-kv260, DPUCVDX8G) the notebook generates an output file name "tvm_dpu_cpu.so"
# Once you setup your edge device, you could copy the compiled file to your
#   acceleration device and run the model with the following variables:
#
# Parameter settings for the run script:
# -f           : Path to the exported TVM compiled model (tvm_dpu_cpu.so in the example)
# -p           : Specify if the TF (Tensorflow) or ONNX model was used for compilation
# --iterations : The number of iterations to run the model
#
# example:
# ./run_external_yolov3.py -f /PATH_TO_DIR/tvm_dpu_cpu.so -p TF --iterations 1
#
##############################################################################
 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Path to TVM library file (.so)", default=FILE_DIR)
    parser.add_argument("--iterations", help="The number of iterations to run.", default=1, type=int)
    parser.add_argument("-p", help="The input model framework (TF, ONNX)", default="TF", type=str)
    args = parser.parse_args()
    file_path = args.f if os.path.isabs(args.f) else os.path.join(os.getcwd(), args.f)
    iterations = args.iterations
    input_name = 'input/input_data:0' if args.p == "ONNX" else 'input/input_data'
    shape_dict = {input_name: [1, 320, 320, 3]}
    run(file_path, shape_dict, iterations)
