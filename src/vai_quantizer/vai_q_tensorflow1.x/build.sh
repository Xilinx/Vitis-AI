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

# cmake args
declare -a args

# parse options
options=$(getopt -a -n 'parse-options' -o h \
		 -l help,clean,fast,abi0,build_with_cpu,conda,type:,cmake-options: \
		 -- "$0" "$@")
[ $? -eq 0 ] || {
    echo "Failed to parse arguments! try --help"
    exit 1
}
eval set -- "$options"
while true; do
    case "$1" in
	-h | --help) show_help=true; break;;
	--clean) clean=true;;
	--fast) fast=true;;
	--abi0) abi0=true;;
	--build_with_cpu) build_with_cpu=true;;
	--conda) conda=true;;
	--type)
	    shift
	    case "$1" in
		release) build_type=Release;;
		debug) build_type=Debug;;
		*) echo "Invalid build type \"$1\"! try --help"; exit 1;;
	    esac
	    ;;
	--cmake-options) shift; args+=($1);;
	--) shift; break;;
    esac
    shift
done

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
  if ${abi0:=false}; then
    args+=(-DABI1=OFF)
  else
    args+=(-DABI1=ON)
  fi

  # set gpu
  if ${build_with_cpu:=false}; then
    echo "Build vai_q_tensorflow with CPU"
    sed -e 's/DEVICE_SED_MARK/cpu/' proto_version.py > version.py
    args+=(-DGPU=OFF)
  else
    args+=(-DGPU=ON)
    echo "Build vai_q_tensorflow with GPU"

    # TODO: find cuda version automaticlly
    # copy cuda include files to tensorflow third party
    if [ ${PYTHON_LIB_PATH} ]; then
      echo "PYTHON_LIB_PATH exist: ${PYTHON_LIB_PATH}"
    else
      PYTHON_LIB_PATH=`python -c 'import site; print(site.getsitepackages()[0])'`
      echo "Found PYTHON_LIB_PATH: ${PYTHON_LIB_PATH}"
    fi
    tf_include_dir=${PYTHON_LIB_PATH}/tensorflow_core/include/
    dst_dir=${tf_include_dir}/third_party/gpus/cuda/include

    cuda_include_dir=/usr/local/cuda-10.0/targets/x86_64-linux/include
    echo "copy include filse to tensorflow third party"
    mkdir -p ${dst_dir}
    cp -r ${cuda_include_dir}/* ${dst_dir}
    sed -e 's/DEVICE_SED_MARK/gpu/' proto_version.py > version.py
  fi


  if [ ${show_help:=false} == true ]; then
    echo "./build.sh [options]"
    echo "    --help                    show help"
    echo "    --clean                   discard build dir before build"
    echo "    --fast                    run make directly"
    echo "    --build_with_cpu          build with cpu"
    echo "    --abi0                    build with ABI=0"
    echo "    --conda                   search lib path in conda env"
    echo "    --type[=TYPE]             build type. VAR {release, debug(default)}"
    echo "    --cmake-options[=OPTIONS] append more cmake options"
    exit 0
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
cp ./version.py ${gen_files}/

GIT_VERSION=$(git rev-parse --short HEAD)
echo "__git_version__ = \"${GIT_VERSION}\"" >> ${gen_files}/version.py

rm -rf pkgs/*

bash ./pip_pkg.sh ./pkgs/ --release

pip uninstall ./pkgs/*.whl -y
pip install ./pkgs/*.whl
