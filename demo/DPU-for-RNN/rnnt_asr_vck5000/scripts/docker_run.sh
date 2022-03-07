#!/bin/bash

# HERE=`dirname $(readlink -f $0)` # Absolute path of current directory
HERE=`pwd -P`

user=`whoami`
uid=`id -u`
gid=`id -g`
GRP=`id -n -g`

#echo "$user $uid $gid"

DOCKER_REPO="xilinx/"

BRAND=vitis-ai-rnn
VERSION=latest

CPU_IMAGE_TAG=${DOCKER_REPO}$BRAND:${VERSION}
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
    --shm-size 1G \
    -${DETACHED}it \
    --rm \
    --network=host \
    $IMAGE_NAME \
    bash
fi

