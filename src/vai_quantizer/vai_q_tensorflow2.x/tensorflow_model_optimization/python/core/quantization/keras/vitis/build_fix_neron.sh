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
# ==============================================================================

#!/usr/bin/env bash

set -e

root=./vai_q_tensorflow
cmake_build_dir=${root}/build
gen_files=${root}/gen_files
mkdir -p ${cmake_build_dir}
mkdir -p $gen_files


if [ ${fast:=false} == true ]; then
  pushd ${cmake_build_dir}
  echo "Skip cmake and run make directly"
  make -j
  popd
else
  os=`lsb_release -a | grep "Distributor ID" | sed 's/^.*:\s*//'`
  os_version=`lsb_release -a | grep "Release" | sed 's/^.*:\s*//'`
  arch=`uname -p`
  # set build type
  if [ ${build_type:=Release} == "Release" ]; then
      args+=(-DDEBUG=OFF)
      target_info=${os}.${os_version}.${arch}.Release
  else
      args+=(-DDEBUG=ON)
      target_info=${os}.${os_version}.${arch}.Debug
  fi

  if [ ${conda:=false} == true ]; then
    install_prefix=${CONDA_PREFIX}
  else
    install_prefix=$HOME/.local/${target_info}
  fi
  echo "set CMAKE_INSTALL_PREFIX=$install_prefix"
  args+=(-DCMAKE_INSTALL_PREFIX=${install_prefix})

  # set abi
  if ${abi1:=false}; then
    args+=(-DABI1=ON)
  else
    args+=(-DABI1=OFF)
  fi

  # set gpu
  if ${build_with_cpu:=false}; then
    args+=(-DGPU=OFF)
    echo "Build vai_q_tensorflow with CPU"
  else
    args+=(-DGPU=ON)
    echo "Build vai_q_tensorflow with GPU"
  fi

  if [ ${show_help:=false} == true ]; then
    echo "./build.sh [options]"
    echo "    --help                    show help"
    echo "    --clean                   discard build dir before build"
    echo "    --fast                    run make directly"
    echo "    --build_with_cpu          build with cpu"
    echo "    --abi1                    build with ABI=1"
    echo "    --conda                   search lib path in conda env"
    echo "    --type[=TYPE]             build type. VAR {release, debug(default)}"
    echo "    --cmake-options[=OPTIONS] append more cmake options"
    exit 1
  else
    if ${clean:=false}; then
      echo "discard build dir before build"
      rm -fr ./build/* ${cmake_build_dir}/* $gen_files/*
    fi
    pushd ${cmake_build_dir}
    echo cmake "${args[@]}" ..
    cmake "${args[@]}" ..
    make -j
    popd
  fi
fi


## copy gen files
cp ${cmake_build_dir}/fix_neuron_op/*.so ${gen_files}/
cp ${cmake_build_dir}/vai_wrapper.py ${gen_files}/
cp ${cmake_build_dir}/_vai_wrapper.so ${gen_files}/
cp ${root}/python/__init__.py ${gen_files}

