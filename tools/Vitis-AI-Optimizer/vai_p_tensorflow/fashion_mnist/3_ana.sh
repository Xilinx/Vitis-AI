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
export TF_CPP_MIN_LOG_LEVEL=1
mkdir -p ${WORKSPACE}

vai_p_tensorflow \
    --action="ana" \
    --input_graph=${BASELINE_GRAPH} \
    --input_ckpt=${BASELINE_CKPT} \
    --eval_fn_path="net.py" \
    --target="acc5" \
    --max_num_batches=50 \
    --workspace=${WORKSPACE} \
    --input_nodes="input_1" \
    --input_node_shapes="1,1,${IMAGE_SIZE},${IMAGE_SIZE}" \
    --output_nodes=\"${OUTPUT_NODES}\" \
    --gpu="0" 2>&1 | tee ana.log
