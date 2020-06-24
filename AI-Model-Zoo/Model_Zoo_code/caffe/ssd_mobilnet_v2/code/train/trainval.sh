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

caffe_xilinx_dir='Path_to_caffe-xilinx/'
if [ ! -d "snapshot" ];then
mkdir snapshot
else
echo "output model dir exist"
fi

phase_type=train
if test $[phase_type] -eq $[train] ;then
   echo "now is for training"
   solver_file='./code/train/solver.prototxt'
   $caffe_xilinx_dir/build/tools/caffe.bin train -solver $solver_file -gpu 2 2>&1 | tee train.log
else
   echo "now is for testing"
   model_file='./float/trainval.prototxt'
   model_weights=./'float/trainval.caffemodel'
   $caffe_xilinx_dir/build/tools/caffe.bin test -model $model_file -weights $model_weights -ssd 11point -iterations 10000 -gpu 0 2>&1 | tee test.log
fi
