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

DOCKER_REPO="xilinx/"
BRAND=vitis-ai
VERSION=1.0.0
DATE="$(date)"

# Final Build Image Tag
GPU_IMAGE_TAG=${DOCKER_REPO}$BRAND:tools-${VERSION}-gpu

docker build --network=host --build-arg VERSION=1.0.0 --build-arg DATE="$(date)" -f Dockerfile.gpu -t $GPU_IMAGE_TAG ./
