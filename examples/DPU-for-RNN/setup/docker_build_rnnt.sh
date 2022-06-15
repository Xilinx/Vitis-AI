#!/usr/bin/env bash
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

DOCKER_REPO="${DOCKER_REPO:-xilinx/}"
VERSION="${VERSION:-latest}"
DATE_RNN="${DATE_RNN:-$(date -I)}"
GIT_HASH_RNN="${GIT_HASH_RNN:-$(git rev-parse --short HEAD)}"
CACHEBUST="${CACHEBUST:-1}"
BASE_IMAGE="${BASE_IMAGE:-xilinx/vitis-ai-gpu:latest}"
VAI_CONDA_CHANNEL="${VAI_CONDA_CHANNEL:-file:///scratch/conda-channel}"
CONDA_PACKAGE_PREFIX="${CONDA_PACKAGE_PREFIX:-https://www.xilinx.com/bin/public/openDownload?filename=}"

# Final Build Image Tag
RNN_IMAGE="${RNN_IMAGE:-${DOCKER_REPO}vitis-ai-rnn:${VERSION}}"

# Refresh the BASE_IMAGE
docker pull ${BASE_IMAGE} || true

docker build --network=host \
        --build-arg VERSION=${VERSION} \
        --build-arg DATE_RNN=${DATE_RNN} \
        --build-arg GIT_HASH_RNN=${GIT_HASH_RNN} \
        --build-arg CACHEBUST=${DATE_STAMP} \
        --build-arg BASE_IMAGE=${BASE_IMAGE} \
        --build-arg VAI_CONDA_CHANNEL=${VAI_CONDA_CHANNEL} \
        --build-arg CONDA_PACKAGE_PREFIX=${CONDA_PACKAGE_PREFIX} \
        -f dockerfiles/RNN.Dockerfile \
        -t ${RNN_IMAGE} \
        ./

# Tag image as :latest
docker tag ${RNN_IMAGE} ${DOCKER_REPO}vitis-ai-rnn:latest
