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
# usage:
#  bash ./code/train/trainval.sh

# Declare $PATH_TO_DATASET_DIR and $PATH_TO_DATASET_LIST
caffe_xilinx_dir=Path_to/caffe-xilinx/
#training and testing
if [ ! -d "snapshot" ];then
mkdir snapshot
else
echo "output model dir exist"
fi

caffe_path="/build/tools/caffe.bin"
caffe_path_docker="/bin/caffe"

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

echo "now is for training"
solver_file='./code/train/solver.prototxt'
caffe_exec caffe train -solver $solver_file -gpu 0
