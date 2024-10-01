#!/bin/bash
# Copyright 2022 Xilinx Inc.

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

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <image>"
    exit 2
fi

if [ -z "$1" ]; then
   echo "Usage: $0 <Vitis_AI_DOCKER_NAME>"
   exit 2
fi


HERE=$(pwd -P) # Absolute path of current directory
user=`whoami`
uid=`id -u`
gid=`id -g`

DOCKER_REPO="xilinx/"

BRAND=vitis-ai
VERSION=latest

CPU_IMAGE_TAG=${DOCKER_REPO}${BRAND}-cpu:${VERSION}
GPU_IMAGE_TAG=${DOCKER_REPO}${BRAND}-gpu:${VERSION}
IMAGE_NAME="$1"
DEFAULT_COMMAND="bash"

if [[ $# -gt 0 ]]; then
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

kfd_driver="$(find /dev -name kfd\*)"
for i in ${kfd_driver} ;
do
    docker_devices+="--device=$i "
done

DOCKER_RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ "$HERE" != "$DOCKER_RUN_DIR" ]; then
  echo "WARNING: Please start 'docker_run.sh' from the Vitis-AI/ source directory";
fi

docker_run_params=$(cat <<-END
    -v /dev/shm:/dev/shm \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -v $DOCKER_RUN_DIR:/vitis_ai_home \
    -v $HERE:/workspace \
    -w /workspace \
    --rm \
    --network=host \
    ${DETACHED} \
    ${RUN_MODE} \
    $IMAGE_NAME \
    $DEFAULT_COMMAND
END
)

##############################

if [[ ! -f ".confirm" ]]; then

    if [[ $IMAGE_NAME == *"gpu"* ]]; then
        arch="gpu"
    elif [[ $IMAGE_NAME == *"rocm"* ]]; then
        arch='rocm'
    else
        arch='cpu'
    fi

prompt_file="./docker/dockerfiles/PROMPT/PROMPT_${arch}.txt"

  sed -n '1, 5p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '5, 15p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '15, 28p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '28, 61p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '62, 224p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '224, 308p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '309, 520p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key
  
  confirm
fi

touch .confirm 
docker pull $IMAGE_NAME 
if [[ $IMAGE_NAME == *"gpu"* ]]; then
  docker run \
    $docker_devices \
    --gpus all \
    $docker_run_params
elif [[ $IMAGE_NAME == *"rocm"* ]]; then
  docker run \
    $docker_devices \
    --group-add=render --group-add video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    $docker_run_params
else
  docker run \
    $docker_devices \
    $docker_run_params
fi
