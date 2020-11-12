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


#!/bin/bash

caffe_xilinx_dir='../../../caffe-xilinx/'
model_type='yolov4'
threshold='0.001'
model_file='../../float/test.prototxt'
model_weights='../../float/trainval.caffemodel'
image_root='../../data/coco2014/val2014'
list_file='../../data/coco/5k.txt'
result_file='../../data/result.json'

yolov4_detect_path="/build/examples/yolo/yolov4_detect.bin"
yolov4_detect_path_docker="/bin/yolov4_detect"

caffe_xilinx_dir_docker="/opt/vitis_ai/conda/envs/vitis-ai-caffe/"
caffe_path() {
  exec_name=$1
  exec_path=$caffe_xilinx_dir$(eval echo '$'"${exec_name}_path")
  if [ ! -f "$exec_path" ]; then
    echo >&2 "$exec_path does not exist, try use path in pre-build docker"
    exec_path=$caffe_xilinx_dir_docker$(eval echo '$'"${exec_name}_path_docker")
  fi
  echo "$exec_path"
}

caffe_exec() {
  exec_path=$(caffe_path "$1")
  shift
  $exec_path "$@"
}

# Test images
caffe_exec yolov4_detect $model_file $model_weights $image_root $list_file \
          -mode eval \
          -confidence_threshold $threshold \
          -out_file $result_file \
          -model_type $model_type 

python eval_coco.py
