#!/bin/bash
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

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <image>"
    exit 2
fi

HERE=$(pwd -P) # Absolute path of current directory
user=`whoami`
uid=`id -u`
gid=`id -g`

DOCKER_REPO="xilinx/"

BRAND=vitis-ai-cpu-wego
VERSION=latest

IMAGE_TAG=${DOCKER_REPO}${BRAND}:${VERSION}
DEFAULT_COMMAND="bash"
IMAGE_NAME=$IMAGE_TAG
USE_GPU=""
if [[ $# -gt 0 ]]; then
  IMAGE_NAME="${1:-$IMAGE_TAG}"
  shift 1;
  DEFAULT_COMMAND="$@"
  if [[ -z "$1" ]]; then
    DEFAULT_COMMAND="bash"
  fi
fi

DETACHED="-it"

xclmgmt_driver="$(find /dev -name xclmgmt\*)"
docker_devices=""
for i in ${xclmgmt_driver} ;
do
  docker_devices+="--device=$i "
done

render_driver="$(find /dev/dri -name renderD\*)"
for i in ${render_driver} ;
do
  docker_devices+="--device=$i "
done

DOCKER_RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ "$PWD" != "$DOCKER_RUN_DIR" ]; then
  echo "WARNING: Please start 'docker_run.sh' from the Vitis-AI/ source directory";
fi

docker_run_params=$(cat <<-END
    -v /dev/shm:/dev/shm \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -v /etc/xbutler:/etc/xbutler \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -e VERSION=$VERSION \
    -v $DOCKER_RUN_DIR:/vitis_ai_home \
    -v $HERE:/workspace \
    -w /workspace \
    --rm \
    --network=host \
    ${DETACHED} \
    ${RUN_MODE} \
    ${USE_GPU} \
    $IMAGE_NAME \
    $DEFAULT_COMMAND
END
)

##############################
echo "[INFO] running docker image: '$IMAGE_NAME'"
docker run \
  $docker_devices \
  $docker_run_params