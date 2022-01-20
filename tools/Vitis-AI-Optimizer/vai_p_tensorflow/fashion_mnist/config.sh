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


WORKSPACE="./pruning"

OUTPUT_NODES="dense_1/Softmax"
LOGITS="dense_1/Softmax:0"

GRAPH_FILE_NAME="fashion_mnist_inf_graph.pbtxt"
BASELINE_DIR="./baseline"
BASELINE_CKPT="${BASELINE_DIR}/model"
BASELINE_GRAPH="${BASELINE_DIR}/${GRAPH_FILE_NAME}"

IMAGE_SIZE=28

SPARSITY=0.2

PRUNED_GRAPH="${WORKSPACE}/sparse_${SPARSITY}.pbtxt"
PRUNED_CKPT="${WORKSPACE}/sparse_${SPARSITY}"

FT_EPOCHS=5
FT_CKPT="${WORKSPACE}/model_${SPARSITY}"

TRANSFORMED_CKPT="${WORKSPACE}/transformed_${SPARSITY}"

FROZEN_PB="${WORKSPACE}/model_${SPARSITY}.pb"
