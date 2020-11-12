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

echo "Preparing dataset..."

DATA_DIR=./data/
DATASET=cityscapes
WEIGHTS=float


echo "Conducting ENet_xilinx Quantization"

export PYTHONPATH=${PWD}:${PYTHONPATH} 
export W_QUANT=1
GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --val_batch_size 4 --data_root $DATA_DIR${DATASET} --input_size 1024 512 --weight  ${WEIGHTS}/final_best.pth --quant_mode calib

CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --val_batch_size 4 --data_root $DATA_DIR${DATASET} --input_size 1024 512 --weight  ${WEIGHTS}/final_best.pth --quant_mode test
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --dump_xmodel --eval --val_batch_size 4 --data_root $DATA_DIR${DATASET} --input_size 1024 512 --weight  ${WEIGHTS}/final_best.pth --quant_mode test

echo "Test finishes!"
echo "======================================================================================="
