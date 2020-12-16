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
#  bash eval.sh

## prepare dataset

echo "please download EDD dataset and put it into \"data/EDD\""
# run test'
CAFFE_PATH='***'
echo " =====> Begin testing....."
python code/test.py  --caffepath $CAFFE_PATH --modelpath ./float/ --prototxt_file pytorch2caffe_mergebn2conv.prototxt --weight_file pytorch2caffe_mergebn2conv.caffemodel --imgpath ./data/EDD/images --savepath ./results/ --num-classes 2

echo " =====> moU evaluation ..."
python code/eval_edd.py --GT_maskDIR ./data/EDD/labels/ --Eval_maskDIR ./results/ --Img_DIR ./results/BE/

echo " =====> If evaluation performance is same as following:"
echo " mean dice=0.8203; mean jaccard=0.7925; F2-score=0.8075"
echo " testing successfully!"


