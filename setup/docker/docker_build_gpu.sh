#!/usr/bin/env bash
# Copyright 2020 Xilinx Inc.
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
VERSION="${VERSION:-`cat docker/VERSION.txt`}"
DOCKERFILE="${DOCKERFILE:-docker/DockerfileGPU}"

BRAND="${BRAND:-vitis-ai-gpu}"
DATE="$(date)"

# Final Build Image Tag
IMAGE_TAG=${DOCKER_REPO}$BRAND:${VERSION}

docker build --network=host --build-arg VERSION=${VERSION} --build-arg CACHEBUST="$(date +%s)" --build-arg DATE="$(date -I)" -f ${DOCKERFILE} -t $IMAGE_TAG ./
