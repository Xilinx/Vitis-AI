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
checkpoint='../../float/checkpoint_resnet20_bn_relu_am_msra_celebrity_80_0.4_0811_26.tar'
testset_images='../../data/test/lfw/lfw_aligned_densebox_new'
testset_list='../../data/test/lfw/lfw.txt'
testset_pairs='../../data/test/lfw/pairs.txt'

export W_QUANT=0
python test_face.py \
    --checkpoint $checkpoint \
    --testset_images $testset_images \
    --testset_list $testset_list \
    --testset_pairs $testset_pairs \
    --batch-size 128 \
    --quant_mode float \
