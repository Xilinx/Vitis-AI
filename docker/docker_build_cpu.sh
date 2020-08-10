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
DOCKERFILE="${DOCKERFILE:-DockerfileCPU/}"

BRAND="${BRAND:-vitis-ai-cpu}"
DATE="$(date)"

# Final Build Image Tag
IMAGE_TAG=${DOCKER_REPO}$BRAND:${VERSION}

sed -n '1, 5p' ./PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key

sed -n '5, 15p' ./PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key

sed -n '15, 24p' ./PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key

sed -n '24, 61p' ./PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key

sed -n '62, 300p' ./PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key

sed -n '300, 308p' ./PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key


confirm() {
  echo -n "Do you agree to the terms and wish to proceed [y/n]? "
  read REPLY
  case $REPLY in
    [Yy]) break ;;
    [Nn]) exit 0 ;;
    *) confirm ;;
  esac
    REPLY=''
}

confirm

docker build --network=host --build-arg VERSION=${VERSION} --build-arg CACHEBUST="$(date +%s)" --build-arg DATE="$(date)" -f ${DOCKERFILE} -t $IMAGE_TAG ./
