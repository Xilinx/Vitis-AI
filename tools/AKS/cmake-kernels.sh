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

#!/bin/bash

# ROOT dir
ROOT=$(dirname "$(realpath $0)")

# Formatting
red=$(tput setaf 1)
green=$(tput setaf 2)
yellow=$(tput setaf 3)
white=$(tput setaf 7)
bold=$(tput bold)
reset=$(tput sgr0)

MSG="[INFO]"
SUC="${green}${bold}[INFO]"
WAR="${yellow}${bold}[WRNG]"
ERR="${red}${bold}[CERR]"
RST="${reset}"

declare -a args
# parse options
options=$(getopt -a -n 'parse-options' -o h \
		 -l help,clean,clean-only,aks-install-prefix: \
		 -- "$0" "$@")
[ $? -eq 0 ] || {
    echo "Failed to parse arguments! try --help"
    exit 1
}
eval set -- "$options"
while true; do
  case "$1" in
	-h | --help                ) show_help=true; break;;
	     --clean               ) clean=true;;
	     --clean-only          ) clean_only=true;;
	     --aks-install-prefix  ) shift; install_prefix=$1;;
	     --) shift; break;;
  esac
  shift
done

# Usage
if [ ${show_help:=false} == true ]; then
  echo -e
  echo -e "$0 [options]"
  echo -e "    --help                 show help"
  echo -e "    --clean                discard previous configs/builds before build"
  echo -e "    --clean-only           discard previous configs/builds"
  echo -e "    --aks-install-prefix   set customized aks install prefix"
  echo -e
  exit 0
fi

args=(-DAKS_INSTALL_PREFIX="${install_prefix}")

# Get all Kernels
KERNELS=$(ls -d -1 kernel_src/*)

ARM_INCOMPATIBLE="kernel_src/caffe_kernel \
	kernel_src/dpuv1_norunner \
	kernel_src/dpuv1_runner \
	kernel_src/classification_preprocess_accel \
	kernel_src/classification_fc_softmax_top_k \
	kernel_src/fully_connected_relu \
	kernel_src/fully_connected_sigmoid \
	kernel_src/optical_flow_fpga \
	kernel_src/yolo_postprocess"
x86_INCOMPATIBLE="kernel_src/dpuv2_runner"
arch=$(uname -m)

# Filter out architecture incompatible kernels
if [[ ${arch} == "x86_64" ]]
then
	INCOMPATIBLE=${x86_INCOMPATIBLE}
elif [[ ${arch} == "aarch64" ]]
then
	INCOMPATIBLE=${ARM_INCOMPATIBLE}
else
	echo "[ERROR] Unknown architecture - ${arch}"
	echo "[ERROR] Supported architectures - x86_64/aarch64"
	exit
fi

a=$(echo ${KERNELS} | tr [:space:] "\n" | sort -u)
b=$(echo ${INCOMPATIBLE} | tr [:space:] "\n" | sort -u)
for i in ${b}
do
	a=${a/$i/}
done
KERNELS=$a

declare -a skipped_kernels
declare -a failed_kernels

# Build all kernels
for kernel in ${KERNELS}; do
  echo -e
  cd ${ROOT}
  # Check if kernel has CMakeLists
  echo -e "-------------------------------------------------------------------"
  echo -e "${bold}${MSG} Kernel Path: ${kernel} ${RST}"
  echo -e "-------------------------------------------------------------------"
  if [ -f "${kernel}/CMakeLists.txt" ]; then
    echo -e "${MSG} ${kernel}/CMakeLists.txt Found. ${RST}"
    echo -e "${MSG} Create Build Dir: ${kernel}/build ${RST}"
  else
    echo -e "${WAR} ${kernel}/CMakeLists.txt Not-Found, Skip Build ${RST}"
    skipped_kernels+=(${kernel})
    continue
  fi

  # Create build directory
  mkdir -p ${kernel}/build && cd ${kernel}/build
  if [ ${clean_only:=false} == true ]; then
    echo -e "${SUC} Clean existing configs ${RST}"
    rm -rf *
    continue
  fi

  if [ ${clean:=false} == true ]; then
    echo -e "${WAR} Clean existing configs ${RST}"
    rm -rf *
  fi

  # Configure and Build Kernel
  echo -e "${MSG} Configure Kernel ... ${RST}"
  cmake "${args[@]}" ..
  if [ $? -eq 0 ]; then
    echo -e "${SUC} Configure Successful ${RST}"
  else
    echo -e
    echo -e "${ERR} Configure Failed for ${kernel} ${RST}"
    failed_kernels+=(${kernel})
    continue
  fi

  echo -e "${MSG} Build Kernel ${RST}"
  cmake --build .

  if [ $? -eq 0 ]; then
    echo -e "${SUC} Build Successful ${RST}"
  else
    echo -e
    echo -e "${ERR} Build Failed for ${kernel} ${RST}"
    failed_kernels+=(${kernel})
    continue
  fi

  # Copy generated libs to root dir
  # echo -e "${MSG} Copy generated libs to ${ROOT}/libs ${RST}"
  cp -P lib*.so* ${ROOT}/libs
done
echo -e "-------------------------------------------------------------------"

if [ ! -z "${skipped_kernels}" ]; then
  echo -e
  echo -e "${WAR} Skipped Kernels ${RST}"
  for sk in ${skipped_kernels[@]}; do
    echo -e "------ ${sk}"
  done
fi

if [ ! -z "${failed_kernels}" ]; then
  echo -e
  echo -e "${ERR} Failed Kernels ${RST}"
  for fk in ${failed_kernels[@]}; do
    echo -e "------ ${fk}"
  done
fi
echo -e
