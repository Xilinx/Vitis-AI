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

WORKSPACE="./pruning"
BASELINE_DIR="./baseline"

if [[ ! -d "${WORKSPACE}" ]]; then
  mkdir -p "${WORKSPACE}"
fi

if [[ ! -d "${BASELINE_DIR}" ]]; then
  mkdir -p "${BASELINE_DIR}"
fi

DATA_DIR="./dataset/cifar10"
SPARSITY=0.3
BASELINE_PATH="${BASELINE_DIR}/model.pth"

FT_EPOCHS=5
GPUS="0"

