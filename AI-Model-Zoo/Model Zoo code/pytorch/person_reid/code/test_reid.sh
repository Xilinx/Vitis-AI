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

#!/usr/bin/env bash
checkpoint_dir=path_to_personreid_model
dataset_dir=../../data

python3 code/test/test.py --config_file='code/test/configs/personreid_market.yml' \
--dataset='market1501' \
--dataset_root=${dataset_dir}'/market1501' \
--load_model=${checkpoint_dir}'/personreid_resnet50.pth' \
--gpu=0 \
 | tee ./test_personreid.log

