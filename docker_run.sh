#!/bin/bash

sed -n '1, 5p' ./docker/PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key

sed -n '5, 15p' ./docker/PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key

sed -n '15, 28p' ./docker/PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key

sed -n '28, 61p' ./docker/PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key

sed -n '62, 224p' ./docker/PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key

sed -n '224, 308p' ./docker/PROMPT.txt
read -n 1 -s -r -p "Press any key to continue..." key


confirm() {
  echo -en "\n\nDo you agree to the terms and wish to proceed [y/n]? "
  read REPLY
  case $REPLY in
    [Yy]) ;;
    [Nn]) exit 0 ;;
    *) confirm ;;
  esac
    REPLY=''
}

confirm


if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <image>"
    exit 2
fi

HERE=$(pwd) # Absolute path of current directory

user=`whoami`
uid=`id -u`
gid=`id -g`

#echo "$user $uid $gid"

DOCKER_REPO="xilinx/"

BRAND=vitis-ai
VERSION=latest

CPU_IMAGE_TAG=${DOCKER_REPO}$BRAND:${VERSION}-cpu
GPU_IMAGE_TAG=${DOCKER_REPO}$BRAND:${VERSION}-gpu
IMAGE_NAME="${1:-$CPU_IMAGE_TAG}"

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

if [[ $IMAGE_NAME == *"sdk"* ]]; then
  docker run \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -v $HERE:/workspace \
    -v /dev/shm:/dev/shm \
    -w /workspace \
    -it \
    --rm \
    --network=host \
    $IMAGE_NAME \
    bash
elif [[ $IMAGE_NAME == *"gpu"* ]]; then
  docker run \
    $docker_devices \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -v $HERE:/workspace \
    -v /dev/shm:/dev/shm \
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
    -e USER=$user -e UID=$uid -e GID=$gid \
    -v /dev/shm:/dev/shm \
    -v $HERE:/workspace \
    -w /workspace \
    -it \
    --rm \
    --network=host \
    $IMAGE_NAME \
    bash
fi

