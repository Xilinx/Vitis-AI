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

export PYTHONPATH=${PWD}:${PYTHONPATH}

CUDA_VISIBLE_DEVICES='0,1' python train/train.py \
        --data-set ChaosCT \
        --gpu 0,1,2 \
        --data-dir '../data/Chaos_CT_processed/CHAOS_Train_Sets/Train_Sets/CT/' \
        --classes-num 2 \
        --input-size 512,512 \
        --random-mirror \
        --lr 1e-2 \
        --weight-decay 5e-4 \
        --batch-size 50 \
        --num-steps 20000 \
        --is-load-imgnet True \
        --ckpt-path '../ckpt_unet_lw/' \
        --pretrain-model-imgnet '../float/unet/model_bs100/ChaosCT_0.9770885167.pth'                                                                           
