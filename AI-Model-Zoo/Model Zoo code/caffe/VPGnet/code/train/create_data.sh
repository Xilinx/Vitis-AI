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
$caffe_xilinx_dir/build/tools/convert_driving_data $PATH_TO_DATASET_DIR $PATH_TO_DATASET_LIST/train_caltech.txt $PATH_TO_DATASET_DIR/LMDB_train
$caffe_xilinx_dir/build/tools/compute_driving_mean $PATH_TO_DATASET_DIR/LMDB_train ./driving_mean_train.binaryproto lmdb
$caffe_xilinx_dir/build/tools/convert_driving_data $PATH_TO_DATASET_DIR $PATH_TO_DATASET_LIST/train_caltech.txt $PATH_TO_DATASET_DIR/LMDB_test
