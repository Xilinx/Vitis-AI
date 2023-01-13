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
DOCKERFILE="${DOCKERFILE:-dockerfiles/ubuntu-vai/vitis-ai-cpu.Dockerfile}"
XRT_URL="${XRT_URL:-https://www.xilinx.com/bin/public/openDownload?filename=xrt_202220.2.14.354_20.04-amd64-xrt.deb}"
XRM_URL="${XRM_URL:-https://www.xilinx.com/bin/public/openDownload?filename=xrm_202220.1.5.212_20.04-x86_64.deb}"
VAI_CONDA_CHANNEL="${VAI_CONDA_CHANNEL:-https://www.xilinx.com/bin/public/openDownload?filename=conda-channel-3.0.tar.gz}"
VAI_WEGO_CONDA_CHANNEL="${VAI_WEGO_CONDA_CHANNEL:-https://www.xilinx.com/bin/public/openDownload?filename=conda-channel-wego-3.0.tar.gz}"
VAI_DEB_CHANNEL="${VAI_DEB_CHANNEL:-https://www.xilinx.com/bin/public/openDownload?filename=vairuntime-3.0.tar.gz}"
SUCCESSFUL_EXIT_STATUS=0
FAILED_EXIT_STATUS=1
SKIP_BUILD_BASE_IMAGE=0


function usage
{
    local rtn=${SUCCESSFUL_EXIT_STATUS}
    echo "usage: $0 -t DOCKER_TYPE  -f FRAMEWORK "
    echo "   ";
    echo "   This script builds Vitis AI dockers"
    echo "   ";
    echo "  -t | --DOCKER_TYPE         : [Required] Valid values:cpu,gpu,rocm";
    echo "  -f | --TARGET_FRAMEWORK    : [Required] The Framework to build. Valid values:" 
    echo "                              For CPU docker: 
                                            Tensorflow 1.15: tf1
					    Tensorflow 2 :   tf2
					    Pytorch:         pytorch";
    echo  "                             For GPU dockers:
                                            TensorFlow 1.15:       tf1
                                            TensorFlow 2 :         tf2
                                            optimizer_tensorflow:  opt_tf1
                                            optimizer_tensorflow2: opt_tf2
                                            optimizer_pytorch:     opt_pytorch
                                            PyTorch:               pytorch"; 
    echo  "                             For ROCM dockers:
                                            optimizer_tensorflow2: opt_tf2
                                            optimizer_pytorch:     opt_pytorch
                                            TensorFlow 2 :         tf2
                                            PyTorch:               pytorch";
    echo "  -h | --help              : This message";
    return ${rtn}
}

# Execute
function execute
{

  add_args=""
  echo "SKIP:${SKIP_BUILD_BASE_IMAGE}, docker type:${DOCKER_TYPE}\n"
 if [[ "$SKIP_BUILD_BASE_IMAGE" == "0" ]];then
     if [[ "$DOCKER_TYPE" == 'cpu' ]];then
         VAI_BASE="ubuntu:20.04"
     fi
     if [[ "$DOCKER_TYPE" == 'gpu' ]];then
         VAI_BASE="nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04"
     fi
     if [[ "$DOCKER_TYPE" == 'rocm' ]]; then
	if [[ $TARGET_FRAMEWORK =~ .*"pytorch"* ]];then
 	    VAI_BASE="rocm/pytorch:rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1"
	    add_args=" --build-arg TARGET_FRAMEWORK=$TARGET_FRAMEWORK "
            BASE_IMAGE="${BASE_IMAGE:-xiinx/vitis-ai-${DOCKER_TYPE}-pytorch-base}"
         else
            VAI_BASE="rocm/tensorflow:rocm5.4.1-tf2.10-dev"
         fi
     fi
     BASE_IMAGE="${BASE_IMAGE:-xiinx/vitis-ai-${DOCKER_TYPE}-base}"
     echo "VAI_BASE: ${VAI_BASE}"

     echo "BUild Base image:${BASE_IMAGE} first"
     buildcmd="docker build  --network=host \
	       --build-arg DOCKER_TYPE=$DOCKER_TYPE \
               --build-arg VAI_BASE=${VAI_BASE} \
               -t ${BASE_IMAGE} $add_args  \
               -f dockerfiles/ubuntu-vai/CondaBase.Dockerfile  \
                .
         " 
    echo "CMD: $buildcmd \n"
    ${buildcmd}
    rtn=$?
    if [[ $rtn -ne 0 ]]; then
     echo "FAILD build base image"
     exit 1
    fi


 fi
 buildcmd="docker build --network=host \
     --build-arg TARGET_FRAMEWORK=$TARGET_FRAMEWORK \
     --build-arg DOCKER_TYPE=$DOCKER_TYPE \
     --build-arg XRT_URL=${XRT_URL} \
     --build-arg XRM_URL=${XRM_URL} \
     --build-arg VAI_BASE=${BASE_IMAGE} \
     --build-arg PETALINUX_URL=${PETALINUX_URL} \
     --build-arg VAI_CONDA_CHANNEL=${VAI_CONDA_CHANNEL} \
     --build-arg VAI_WEGO_CONDA_CHANNEL=${VAI_WEGO_CONDA_CHANNEL} \
     --build-arg VAI_DEB_CHANNEL=${VAI_DEB_CHANNEL} \
     --build-arg VERSION=${VERSION} \
     --build-arg GIT_HASH=`git rev-parse --short HEAD` \
     --build-arg CACHEBUST=$(date +%s) \
     --build-arg BUILD_DATE=$(date -I) \
     -f ${DOCKERFILE} -t $IMAGE_TAG ./ "
 echo "$buildcmd"
 $buildcmd
 docker tag ${IMAGE_TAG} ${IMAGE_LATEST_TAG}
 rtn=$?
 return ${rtn}
}

# Prereq
function prereq
{
    local rtn=${SUCCESSFUL_EXIT_STATUS}
    # passed
    return ${rtn}
}

# Validate Arguments
function validate_args
{
    local rtn=${SUCCESSFUL_EXIT_STATUS}
    echo -ne "Validating Arguments... \n"
    echo "Your inputs: Docker-Type:$DOCKER_TYPE, FrameWork:$TARGET_FRAMEWORK"
    # Is DIRECTORY set?
    if [ -z "${TARGET_FRAMEWORK}" ]; then
        echo "Please specify FREAMEWORK to BUILD!"
        rtn=${FAILED_EXIT_STATUS}
    else
	if [[ "$DOCKER_TYPE" == 'cpu' ]]; then
          if [[ "$TARGET_FRAMEWORK" != "tf1" && "$TARGET_FRAMEWORK" != 'tf2' &&  "$TARGET_FRAMEWORK" != 'pytorch' ]]; then
	    echo "Error: For CPU docker, Validate value for TARGET_FRAMEWORK are:"
	    echo "    tf1,tf2,pytorch"
	     rtn=${FAILED_EXIT_STATUS}
          fi
        elif [[ "$DOCKER_TYPE" == 'gpu' ]]; then
           if [[ "$TARGET_FRAMEWORK" != "tf1" && "$TARGET_FRAMEWORK" != 'tf2' && "$TARGET_FRAMEWORK" != 'pytorch' && "$TARGET_FRAMEWORK" != 'opt_tf1' && "$TARGET_FRAMEWORK" != 'opt_tf2' && "$TARGET_FRAMEWORK" != 'opt_pytorch' ]]; then
            echo "Error: For Nvidia GPU  docker, Validate value for TARGET_FRAMEWORK are:"
            echo "    tf1,tf2,pytorch,opt_tf1,opt_tf2, opt_pytorch"
             rtn=${FAILED_EXIT_STATUS}
          fi

        elif [[ "$DOCKER_TYPE" == 'rocm' ]]; then
          if [[ "$TARGET_FRAMEWORK" != "opt_tf2" && "$TARGET_FRAMEWORK" != 'opt_pytorch' &&  "$TARGET_FRAMEWORK" != 'pytorch' &&  "$TARGET_FRAMEWORK" != 'tf2' ]]; then
            echo "Error: For ROCM docker, Validate value for TARGET_FRAMEWORK are:"
            echo "    tf2,pytorch,opt_pytorch,opt_tf2"
             rtn=${FAILED_EXIT_STATUS}
          fi
        fi
    fi

    # pass?
    if [ ${rtn} -eq "${FAILED_EXIT_STATUS}" ]; then
	    usage
    else
        echo "passed!"
    fi
    return ${rtn}
}

# Parse Arguments
function parse_args
{
    local rtn=${SUCCESSFUL_EXIT_STATUS}
    # positional args
    args=()

    # named args
    while [ "$1" != "" ]; do
        case "$1" in
           -t| --DOCKER_TYPE)
                shift
                DOCKER_TYPE=$1
                ;;
            -f|--TARGET_FRAMEWORK)
                shift
                TARGET_FRAMEWORK=$1
                ;;
            -b| --BASE_IMAGE)
                shift
                BASE_IMAGE=$1
                ;;
            -s| --SKIP_BUILD_BASE_IMAGE)
                shift
                SKIP_BUILD_BASE_IMAGE=1
                ;;
            -h | --help )
                usage
                exit
                ;;
            * )
                usage
                exit 1
        esac
        shift
    done
    return ${rtn}
}

# Main
function main
{
    local rtn=${SUCCESSFUL_EXIT_STATUS}
    parse_args "$@"
    rtn=$?
    if [ "${rtn}" -eq "${SUCCESSFUL_EXIT_STATUS}" ]; then
        validate_args
        rtn=$?
    fi
    IMG_FW=""
    case $TARGET_FRAMEWORK in
        tf1) IMG_FW="tensorflow";;
        tf2) IMG_FW="tensorflow2";;
        pytorch) IMG_FW="pytorch";; 
        opt_tf1) IMG_FW="opt-tensorflow";;
        opt_tf2) IMG_FW="opt-tensorflow2";;
        opt_pytorch) IMG_FW="opt-pytorch";; 
    esac

    BRAND="${BRAND:-vitis-ai-${IMG_FW}-${DOCKER_TYPE}}"
    # Final Build Image Tag
    IMAGE_TAG=${DOCKER_REPO}${BRAND}:${VERSION}
    IMAGE_LATEST_TAG=${DOCKER_REPO}${BRAND}:latest

    if [ "${rtn}" -eq "${SUCCESSFUL_EXIT_STATUS}" ]; then
        prereq
        rtn=$?
    fi

    if [ "${rtn}" -eq "${SUCCESSFUL_EXIT_STATUS}" ]; then

        prompt_file="./dockerfiles/PROMPT/PROMPT_${DOCKER_TYPE}.txt"

        sed -n '1, 5p' $prompt_file #./dockerfiles/PROMPT/PROMPT_cpu.txt # $prompt_file
        sed -n '5, 15p' $prompt_file
        read -n 1 -s -r -p "Press any key to continue..." key
        sed -n '15, 28p' $prompt_file
        sed -n '28, 61p' $prompt_file
        read -n 1 -s -r -p "Press any key to continue..." key
        sed -n '62, 224p' $prompt_file
        read -n 1 -s -r -p "Press any key to continue..." key
        sed -n '224, 308p' $prompt_file
        read -n 1 -s -r -p "Press any key to continue..." key
        sed -n '308, 500p' $prompt_file
        read -n 1 -s -r -p "Press any key to continue..." key

       confirm() {                                                                                                                                                                                                        echo -en "\n\nDo you agree to the terms and wish to proceed [y/n]? "
           read REPLY
           case $REPLY in
               [Yy]) ;;
               [Nn]) exit 0 ;;
               *) confirm ;;
           esac
          REPLY=''
       }

    confirm

    execute
    rtn=$?
  fi
  return ${rtn}
}

main "$@";

