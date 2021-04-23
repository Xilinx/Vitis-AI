#!/bin/bash

# Copyright 2021 Xilinx Inc.
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

export TVM_VAI_HOME=$(pwd)
export TVM_HOME="${TVM_VAI_HOME}"/tvm
export PYXIR_HOME="${TVM_VAI_HOME}"/pyxir

if [ -d "${TVM_HOME}" ]; then
  rm -rf ${TVM_HOME}
fi
if [ -d "${PYXIR_HOME}" ]; then
  rm -rf ${PYXIR_HOME}
fi

# CREATE SWAP SPACE
if [ ! -f "/swapfile" ]; then
  fallocate -l 4G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo "/swapfile swap swap defaults 0 0" > /etc/fstab
else
  echo "Couldn't allocate swap space as /swapfile already exists"
fi

# INSTALL DEPENDENCIES
if ! command -v h5cc &> /dev/null; then
  cd /tmp && \
    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.7/src/hdf5-1.10.7.tar.gz && \
    tar -zxvf hdf5-1.10.7.tar.gz && \
    cd hdf5-1.10.7 && \
    ./configure --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf hdf5-1.10.7*
fi

cd ${TVM_VAI_HOME}

pip3 install Cython==0.29.23 h5py==2.10.0 pillow

# DOWNLOAD PYXIR AND TVM
git clone --recursive --branch v0.2.0 --single-branch https://github.com/Xilinx/pyxir.git "${PYXIR_HOME}"
git clone --recursive --single-branch https://github.com/apache/tvm.git "${TVM_HOME}" &&\
    cd ${TVM_HOME} && git checkout cc7f529

# BUILD PYXIR FOR EDGE
cd "${PYXIR_HOME}"
sudo python3 setup.py install --use_vai_rt_dpuczdx8g --use_dpuczdx8g_vart

# BUILD TVM
cd "${TVM_HOME}"
mkdir "${TVM_HOME}"/build
cp "${TVM_HOME}"/cmake/config.cmake "${TVM_HOME}"/build/
cd "${TVM_HOME}"/build && echo set\(USE_VITIS_AI ON\) >> config.cmake && echo set\(USE_LLVM OFF\) >> config.cmake && cmake .. && make tvm_runtime -j$(nproc)
cd "${TVM_HOME}"/python && pip3 install --no-deps  -e .