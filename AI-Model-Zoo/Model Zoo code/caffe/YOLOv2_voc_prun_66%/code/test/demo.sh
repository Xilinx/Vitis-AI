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
#Merge Convolution + BatchNorm + (scale) --> Convolution 
#merge prototxt file
$caffe_xilinx_dir/build/tools/convert_model merge -model_in $model_file -model_out $model_file.nobn
#merge caffemodel file
$caffe_xilinx_dir/build/tools/convert_model merge -weights_in $model_weights -weights_out $model_weights.nobn
# Test images
$caffe_xilinx_dir/build/examples/yolo/yolo_detect.bin $model_file.nobn $model_weights.nobn $list_file \
          -confidence_threshold $threshold \
          -classes $classes \
          -anchorCnt $anchor_pair \
          -out_file $result_file \
          -model_type $model_type \
          -biases $biases 
