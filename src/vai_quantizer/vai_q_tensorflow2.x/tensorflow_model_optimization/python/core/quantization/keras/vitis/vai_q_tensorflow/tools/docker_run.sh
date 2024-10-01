#!/bin/bash
# Copyright 2019 Xilinx Inc.

# HERE=`dirname $(readlink -f $0)` # Absolute path of current directory
HERE=`pwd -P`

user=`whoami`
uid=`id -u`
gid=`id -g`

#echo "$user $uid $gid"

DOCKER_REPO="xdock.xilinx.com/"

BRAND=vitis-ai
VERSION=${VERSION:-latest}

usage() {
    cat >&2 <<EOF
Usage: $0 COMMAND

Commands:
  [ cpu | gpu ] [-d] [-X]
EOF
    exit 1
}

IMAGE_TYPE="cpu"

command=$1;
case "${command}" in
    cpu) IMAGE_TYPE="cpu"; echo "CPU"; shift ;;
    gpu) IMAGE_TYPE="gpu"; echo "GPU"; shift ;;
    -d) ;;
    -X) ;;
    -h) usage ;;
    *) IMAGE_TYPE="cpu" ;;
esac

# -it to run Docker container in interactive mode, -d for detached
DETACHED="-it"
RUN_MODE=""
GUI_ARGS=" -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp/.Xauthority-$user:/tmp/.Xauthority"
IMAGE_NAME=${DOCKER_REPO}${BRAND}-${IMAGE_TYPE}:${VERSION}

#eval set -- "${options}"
#for opt in ${options}; do
#    case "$opt" in
#    -d ) DETACHED="-d"; shift 1 ;;
#    -X ) RUN_MODE=${GUI_ARGS}; shift 1 ;;
#    esac
#done
PREDOCKER=""
POSTDOCKER=""
if [ "$1" == "-d" ]; then
    DETACHED="-d"; shift 1 ;
fi
if [ "$1" == "-X" ]; then
    PREDOCKER="cp -f $HOME/.Xauthority /tmp/.Xauthority-$user;chmod -R a+rw /tmp/.Xauthority-$user"
    POSTDOCKER="rm -Rf /tmp/.Xauthority-$user"
    RUN_MODE=${GUI_ARGS}; shift 1 ;
fi
if [ $# -ne 0 ]; then
    IMAGE_NAME=$1;
fi

#IMAGE_NAME="${1:$DOCKER_REPO$BRAND:$VERSION-$IMAGE_TYPE}"
#CPU_IMAGE_TAG=${DOCKER_REPO}$BRAND:${VERSION}-cpu
#GPU_IMAGE_TAG=${DOCKER_REPO}$BRAND:${VERSION}-gpu
#IMAGE_NAME="${1:-$CPU_IMAGE_TAG}"
#DETACHED="${2:-""}"


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
  cmd="docker run \
    $docker_devices \
    -v /dev/shm:/dev/shm \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -e VERSION=$VERSION \
    -v $HERE:/workspace \
    -w /workspace \
    --rm \
    --gpus all \
    --network=host \
    ${DETACHED} \
    ${RUN_MODE} \
    $IMAGE_NAME \
    bash"
else
  cmd="docker run \
    $docker_devices \
    -v /dev/shm:/dev/shm \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -e VERSION=$VERSION \
    -v $HERE:/workspace \
    -w /workspace \
    --rm \
    --network=host \
    ${DETACHED} \
    ${RUN_MODE} \
    $IMAGE_NAME \
    bash"
fi

eval $PREDOCKER
eval $cmd
eval $POSTDOCKER
