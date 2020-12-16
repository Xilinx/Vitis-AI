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
set -e

WEIGHT_DIR=./float/
MODEL_NAME=erfnet_cbr
SAVE_DIR=results_visulization_${MODEL_NAME}
echo 'perform testing...'

export PYTHONPATH=${PWD}:${PYTHONPATH}

CUDA_VISIBLE_DEVICES=0 python code/test.py  --arch cbr --input_size 512,1024 --img_path ./data/cityscapes_20cls/val_images/ --num_classes 20 --weight_file ${WEIGHT_DIR}/${MODEL_NAME}/model_weights.h5 --save_path $SAVE_DIR 

echo 'perform evaluation...'
python code/utils/evaluate_miou.py --gt ./data/cityscapes_20cls/val_masks/ --result $SAVE_DIR

