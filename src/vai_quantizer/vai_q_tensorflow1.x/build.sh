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


## build *.so
if [ $1 == "fast" ]; then
  # fast build, skip cmake so that will not rebuild all
  pushd ${cmake_build_dir}
  make -j
  popd
elif [ $1 == "CPU" ] || [ $1 == "cpu" ]; then
  echo "Build vai_q_tensorflow without GPU"
  rm -fr ./build/*
  rm -fr ${cmake_build_dir}/*
  rm -fr $gen_files/*
  sed -e 's/DEVICE_SED_MARK/cpu/' proto_version.py > version.py
  pushd ${cmake_build_dir}
  if [ "$2" = ""  ]; then
    echo "ABI is not set, using ABI=1 as default"
    ABI=1
  else
    ABI=$2
  fi
  cmake -DGPU=OFF .. -DABI1=$ABI
  make -j
  popd
else
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

  rm -fr ./build/*
  rm -fr ${cmake_build_dir}/*
  rm -fr $gen_files/*
  sed -e 's/DEVICE_SED_MARK/gpu/' proto_version.py > version.py
  pushd ${cmake_build_dir}
  if [ "$2" = ""  ]; then
    echo "ABI is not set, using ABI=0 as default"
    ABI=0
  else
    ABI=$2
  fi
  cmake .. -DABI1=$ABI
  # cmake -DDEBUG=ON ..
  # cmake -DGPU=ON ..
  # cmake -DGPU=OFF ..
  make -j
  popd
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
