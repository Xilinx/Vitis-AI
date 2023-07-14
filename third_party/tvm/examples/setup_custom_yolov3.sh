#!/bin/bash

# Copyright 2021 Xilinx Inc.
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

# DOWNLOAD AND SETUP CUSTOM YOLOV3 MODEL
. $VAI_ROOT/conda/etc/profile.d/conda.sh  
conda activate vitis-ai-tensorflow

if [ ! -d "/tmp/tensorflow-yolov3" ]; then
    cd /tmp/
    git clone https://github.com/YunYang1994/tensorflow-yolov3 
    cd tensorflow-yolov3
    pip install easydict --user 
    cd checkpoint 
    wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz 
    tar -xvf yolov3_coco.tar.gz
    cd .. 
    python convert_weight.py
    python freeze_graph.py
    sed -i 's/.\//\/tmp\/tensorflow-yolov3\//' ./core/config.py
    # CONVERT TENSORFLOW MODEL TO ONNX
    pip install numpy==1.16.6 --user
    pip install onnx --user 
    cd "${TVM_VAI_HOME}"/tensorflow-yolov3
    git clone https://github.com/onnx/tensorflow-onnx.git
    cd tensorflow-onnx && python setup.py install --user && cd ..
    python3 -m tf2onnx.convert --input ./yolov3_coco.pb --inputs input/input_data:0[1,320,320,3] --outputs pred_sbbox/concat_2:0 --output tf_yolov3_converted.onnx
fi