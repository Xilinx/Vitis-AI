
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

WEIGHT_DIR=./float/
MODEL_NAME=erfnet_cbr

CUDA_VISIBLE_DEVICES=0,1,2,3 python code/train.py \
  --arch cbr \
  --data_path ./data/cityscapes_20cls/ \
  --num_classes 20 \
  --batch_size 5  \
  --resume ${WEIGHT_DIR}/${MODEL_NAME}/model_weights.h5 \
  --quantize true \
  --quantize_output_dir ./quantized/ \
