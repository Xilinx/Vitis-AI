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

caffe_xilinx_dir='PATH_To_caffe-xilinx'
threshold='0.3'
labels='background,person,rider,car,truck,bus,train,motor,bike,sign,light'
model_file='./quantized/fix_test.prototxt'
model_weights='./quantized/fix_train_test.caffemodel'
list_file='./code/test/images.txt'
result_file='./code/test/result.txt'

ssd_detect_path="/build/examples/ssd/ssd_detect.bin"
ssd_detect_path_docker="/bin/ssd_detect"

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

caffe_exec ssd_detect $model_file $model_weights $list_file \
          -confidence_threshold $threshold \
          -out_file $result_file \
          -labels $labels
