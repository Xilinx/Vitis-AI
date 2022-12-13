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

set -e

BASELINE_CKPT="baseline/model"
PRUNED_DIR="pruned"
RATIO=0.3

export TF_CPP_MIN_LOG_LEVEL=1

python mnist_convnet.py \
    --prune ${RATIO} \
    --pretrained "${BASELINE_CKPT}" \
    --save_path "${PRUNED_DIR}/model_${RATIO}"

RATIO=0.8
python mnist_convnet.py \
    --prune ${RATIO} \
    --pretrained "${PRUNED_DIR}/model_0.3"\
    --save_path "${PRUNED_DIR}/model_${RATIO}"
