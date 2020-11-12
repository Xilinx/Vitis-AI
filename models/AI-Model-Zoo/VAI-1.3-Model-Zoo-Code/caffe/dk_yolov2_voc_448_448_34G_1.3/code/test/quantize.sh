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

caffe_xilinx_dir=../../../caffe-xilinx/

echo "now is for quantizing"
MODEL_PATH=../../float/quantize.prototxt
WEIGHT_PATH=../../float/trainval.caffemodel
$caffe_xilinx_dir/build/tools/vai_q quantize --model $MODEL_PATH --weights $WEIGHT_PATH --keep_fixed_neuron --calib_iter 64
