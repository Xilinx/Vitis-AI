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
		 -l help,clean,clean-only,aks-install-prefix:,type: \
		 -- "$0" "$@")
[ $? -eq 0 ] || {
    echo "Failed to parse arguments! try --help"
    exit 1
}
eval set -- "$options"
while true; do
  case "$1" in
    --help | -h           ) show_help=true; break;;
    --clean               ) clean=true;;
    --clean-only          ) clean_only=true;;
    --aks-install-prefix  ) shift; install_prefix=$1;;
    --type)
      shift
      case "$1" in
        release           ) build_type=Release;;
        debug             ) build_type=Debug;;
        *) echo "Invalid build type \"$1\"! try --help"; exit 1;;
      esac;;
    --) shift; break;;
  esac
  shift
done

# Usage
if [ ${show_help:=false} == true ]; then
  echo -e
  echo -e "$0 "
  echo -e "    --help                 show help"
  echo -e "    --clean                discard previous configs/builds before build"
  echo -e "    --clean-only           discard previous configs/builds"
  echo -e "    --type                 set build type [release (Default), debug]"
  echo -e "    --aks-install-prefix   set customized aks install prefix"
  echo -e
  exit 0
fi

args=(-DAKS_INSTALL_PREFIX="${install_prefix}")
# set build type
args+=(-DCMAKE_BUILD_TYPE=${build_type:="Release"})
args+=(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

EXAMPLES=${ROOT}/src

# Check if src has CMakeLists
echo -e "-------------------------------------------------------------------"
echo -e "${bold}${MSG} Path: ${EXAMPLES} ${RST}"
echo -e "-------------------------------------------------------------------"
if [ -f "${EXAMPLES}/CMakeLists.txt" ]; then
  echo -e "${MSG} ${EXAMPLES}/CMakeLists.txt Found! ${RST}"
  echo -e "${MSG} Create Build Dir: ${EXAMPLES}/build ${RST}"
else
  echo -e "${ERR} ${EXAMPLES}/CMakeLists.txt Not-Found, Stopping! ${RST}"
  echo -e
  exit 1
fi

# Create build directory
mkdir -p ${EXAMPLES}/build && cd ${EXAMPLES}/build
if [ ${clean_only:=false} == true ]; then
  echo -e "${SUC} Clean existing configs ${RST}"
  rm -rf *
  rm -rf ../bin
  echo -e
  exit 0
elif [ ${clean:=false} == true ]; then
  echo -e "${WAR} Clean existing configs ${RST}"
  rm -rf *
  rm -rf ../bin
fi

# Configure and Build
echo -e "${MSG} Configure Examples ... ${RST}"
cmake "${args[@]}" ..

if [ $? -eq 0 ]; then
  echo -e "${SUC} Configure Successful! ${RST}"
else
  echo -e
  echo -e "${ERR} Configure Failed! ${RST}"
  echo -e
  exit 1
fi

echo -e "${MSG} Build Examples ... ${RST}"
cmake --build .

if [ $? -eq 0 ]; then
  echo -e "${SUC} Build Successful! ${RST}"
else
  echo -e
  echo -e "${ERR} Build Failed! ${RST}"
  echo -e
  exit 1
fi
echo -e
