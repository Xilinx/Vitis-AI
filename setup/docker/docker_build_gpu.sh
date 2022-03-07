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
VERSION="${VERSION:-`cat dockerfiles/VERSION.txt`}"
DOCKERFILE="${DOCKERFILE:-dockerfiles/vitis-ai-gpu.Dockerfile}"
XRT_URL="${XRT_URL:-https://www.xilinx.com/bin/public/openDownload?filename=xrt_202120.2.12.427_18.04-amd64-xrt.deb}"
XRM_URL="${XRM_URL:-https://www.xilinx.com/bin/public/openDownload?filename=xrm_202120.1.3.29_18.04-x86_64.deb}"
PETALINUX_URL="${PETALINUX_URL:-https://www.xilinx.com/bin/public/openDownload?filename=sdk-2021.2.0.0.sh}"
VAI_CONDA_CHANNEL="${VAI_CONDA_CHANNEL:-https://www.xilinx.com/bin/public/openDownload?filename=conda-channel_2.0.0.1103-02.tar.gz}"


BRAND="${BRAND:-vitis-ai-gpu}"
DATE="$(date)"

# Final Build Image Tag
IMAGE_TAG=${DOCKER_REPO}${BRAND}:${VERSION}
IMAGE_LATEST_TAG=${DOCKER_REPO}${BRAND}:latest
IMAGE_MINOR_TAG=${DOCKER_REPO}${BRAND}:2.0.0
IMAGE_MAJOR_TAG=${DOCKER_REPO}${BRAND}:2.0

docker build --network=host --build-arg VERSION=${VERSION} --build-arg GIT_HASH=`git rev-parse --short HEAD` --build-arg CACHEBUST="$(date +%s)" --build-arg DATE="$(date -I)" -f ${DOCKERFILE} -t ${IMAGE_TAG} ./
docker tag ${IMAGE_TAG} ${IMAGE_MINOR_TAG}
docker tag ${IMAGE_TAG} ${IMAGE_MAJOR_TAG}
docker tag ${IMAGE_TAG} ${IMAGE_LATEST_TAG}
