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

# prepare data

export PYTHONPATH=${PWD}:${PYTHONPATH}

CUDA_VISIBLE_DEVICES=0,1,2,3 python code/train.py --arch cbr --data_path ./data/cityscapes_20cls/ --num_classes 20 --batch_size 32  --learning_rate 1e-2 --epochs 200  --ckpt_path ./checkpoint/erfnet_cbr/
