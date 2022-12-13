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
source config.sh

vai_p_tensorflow \
    --action=prune \
    --input_graph=${BASELINE_GRAPH} \
    --input_ckpt=${BASELINE_CKPT} \
    --output_graph=${PRUNED_GRAPH} \
    --output_ckpt=${PRUNED_CKPT} \
    --workspace=${WORKSPACE} \
    --input_nodes="input_1" \
    --input_node_shapes="1,1,${IMAGE_SIZE},${IMAGE_SIZE}" \
    --output_nodes="${OUTPUT_NODES}" \
    --sparsity=${SPARSITY} \
    --gpu="0"

python -u train.py \
    --pruning=True \
    --pretrained=${PRUNED_CKPT} \
    --epochs=${FT_EPOCHS} \
    --ckpt_path=${FT_CKPT}
