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

# PART OF THIS FILE AT ALL TIMES.

#!bin/sh

gpu=0,1
date=$(date +%m%d%H)
method=${arch}${date}
imgroot="../../data/train/ms_glint"
train_list=$imgroot/lists/msra_celebrity.txt
num_classes=180855
pretrained='../../float/checkpoint_resnet20_bn_relu_am_msra_celebrity_80_0.4_0811_26.tar'
testset_images='../../data/test/lfw/lfw_aligned_densebox_new'
testset_list='../../data/test/lfw/lfw.txt'
testset_pairs='../../data/test/lfw/pairs.txt'

log="log-"${method}".log"

CUDA_VISIBLE_DEVICES=$gpu python train_face.py \
    --num_classes $num_classes \
    --root_path $imgroot \
    --train_list $train_list \
    --pretrained $pretrained \
    --testset_images $testset_images \
    --testset_list $testset_list \
    --testset_pairs $testset_pairs \
    --prefix $method \
    --log $log 
