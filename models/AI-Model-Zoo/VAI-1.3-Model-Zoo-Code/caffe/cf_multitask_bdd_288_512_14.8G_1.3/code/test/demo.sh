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
caffe_xilinx_dir='PathTo_caffe_xilinx'
threshold='0.005'
labels='background,person,car,truck,bus,bike,sign,light'
model_file='./float/test.prototxt'
model_weights='./float/trainval.caffemodel'
list_file='./code/test/demo_image.txt'
result_file='./code/test/result.txt'

seg_detect_ssd_path="/build/examples/ssd/seg_detect_ssd.bin"
seg_detect_ssd_path_docker="/bin/seg_detect_ssd"

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

mkdir -p seg_result/code/train/images

caffe_exec seg_detect_ssd $model_file $model_weights $list_file \
          -confidence_threshold $threshold \
          -out_file $result_file \
          -labels $labels
