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

# Declare $PATH_TO_DATASET_DIR and $PATH_TO_DATASET_LIST
caffe_xilinx_dir=Path_to/caffe-xilinx/
PATH_TO_DATASET_DIR=./code/train/images/
PATH_TO_DATASET_LIST=./code/train/images/

convert_driving_data_path="/build/tools/convert_driving_data"
convert_driving_data_path_docker="/bin/convert_driving_data"
compute_driving_mean_path="/build/tools/compute_driving_mean"
compute_driving_mean_path_docker="/bin/compute_driving_mean"

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

caffe_exec convert_driving_data $PATH_TO_DATASET_DIR $PATH_TO_DATASET_LIST/train_caltech.txt $PATH_TO_DATASET_DIR/LMDB_train
caffe_exec compute_driving_mean $PATH_TO_DATASET_DIR/LMDB_train ./driving_mean_train.binaryproto lmdb
caffe_exec convert_driving_data $PATH_TO_DATASET_DIR $PATH_TO_DATASET_LIST/train_caltech.txt $PATH_TO_DATASET_DIR/LMDB_test
