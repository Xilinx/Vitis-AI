#
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
#

#!/bin/bash

# HERE=`dirname $(readlink -f $0)` # Absolute path of current directory
HERE=`pwd -P`

user=`whoami`
uid=`id -u`
gid=`id -g`
GRP=`id -n -g`

#echo "$user $uid $gid"

DOCKER_REPO="xdock.xilinx.com/"

BRAND=vitis-ai_dev
VERSION=${VERSION:latest}

CPU_IMAGE_TAG=${DOCKER_REPO}$BRAND:${VERSION}-cpu
GPU_IMAGE_TAG=${DOCKER_REPO}$BRAND:${VERSION}-gpu
IMAGE_NAME="${1:-$CPU_IMAGE_TAG}"
DETACHED="${2:-""}"


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

##############################

if [[ $IMAGE_NAME == *"gpu"* ]]; then
  docker run \
    $docker_devices \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -e USER=$user -e UID=$uid -e GID=$gid -e GRP=$GRP\
    -v $HERE:/workspace \
    -w /workspace \
    -it \
    --rm \
    --runtime=nvidia \
    --network=host \
    $IMAGE_NAME \
    bash
else
  docker run \
    $docker_devices \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -e USER=$user -e UID=$uid -e GID=$gid -e GRP=$GRP \
    -v $HERE:/workspace \
    -w /workspace \
    -${DETACHED}it \
    --rm \
    --network=host \
    $IMAGE_NAME \
    bash
fi
