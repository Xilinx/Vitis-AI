#!/usr/bin/env bash

# Copyright 2021 Xilinx Inc.
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

source config.sh

METHOD="iterative"
SAVEDIR=${WORKSPACE}/${METHOD}

if [[ ! -d "${SAVEDIR}" ]]; then
  mkdir -p "${SAVEDIR}"
fi

sparsity_ratios=(0.1 0.2 ${SPARSITY})

for i in ${sparsity_ratios[*]}; do
echo "sparsity=${i}" 
python sparse_model_train.py --gpus ${GPUS} \
  --lr 1e-3 \
  --epochs 5 \
  --sparsity ${i} \
  --pretrained ${BASELINE_PATH} \
  --save_dir ${SAVEDIR} \
  --data_dir ${DATA_DIR} \
  --num_workers 8 \
  --batch_size 128 \
  --weight_decay 1e-4 \
  --momentum 0.9
done
