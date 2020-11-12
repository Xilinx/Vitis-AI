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

caffe_xilinx_dir='PATH_To_caffe_deehi'
model_type='yolov2'
threshold='0.005'
classes='20'
anchor_pair='5'
biases='1.3221,1.73145,3.19275,4.00944,5.05587,8.09892,9.47112,4.84053,11.2364,10.0071'
model_file='./float/test.prototxt'
model_weights='./float/trainval.caffemodel'
list_file='./code/test/images.txt'
result_file='./code/test/result.txt'

convert_model_path="/build/tools/convert_model"
convert_model_path_docker="/bin/convert_model"
yolo_detect_path="/build/examples/yolo/yolo_detect.bin"
yolo_detect_path_docker="/bin/yolo_detect"

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

#Merge Convolution + BatchNorm + (scale) --> Convolution 
#merge prototxt file
caffe_exec convert_model merge -model_in $model_file -model_out $model_file.nobn
#merge caffemodel file
caffe_exec convert_model merge -weights_in $model_weights -weights_out $model_weights.nobn
# Test images
caffe_exec yolo_detect $model_file.nobn $model_weights.nobn $list_file \
          -confidence_threshold $threshold \
          -classes $classes \
          -anchorCnt $anchor_pair \
          -out_file $result_file \
          -model_type $model_type \
          -biases $biases 
