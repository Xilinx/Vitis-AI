#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# DOWNLOAD TVM REPO AND APPLY PATCH

export TVM_VAI_HOME=$(pwd)
export TVM_HOME="${TVM_VAI_HOME}"/tvm
export PYXIR_HOME="${TVM_VAI_HOME}"/pyxir

# DOWNLOADS
git clone --recursive --branch v0.1.3 https://github.com/Xilinx/pyxir.git "${PYXIR_HOME}"
git clone --recursive https://github.com/apache/incubator-tvm.git "${TVM_HOME}"

 
# DOWNLOAD REQUIRED PYTHON PACKAGES
pip3 install cffi cython progressbar h5py==2.8.0
 
# BUILD PYXIR FOR EDGE
cd "${PYXIR_HOME}"
sudo python3 setup.py install --use_vai_rt_dpuczdx8g

# BUILD TVM
cd "${TVM_HOME}"
mkdir "${TVM_HOME}"/build
cp "${TVM_HOME}"/cmake/config.cmake "${TVM_HOME}"/build/
cd "${TVM_HOME}"/build && echo set\(USE_VITIS_AI ON\) >> config.cmake && cmake .. && make tvm_runtime -j$(nproc)
cd "${TVM_HOME}"/python && pip3 install -e .


DISTRIBUTION=`lsb_release -i -s`
if ! [[ "$DISTRIBUTION" == "pynqlinux" ]]; then
    echo " WARNING: You are using the Petalinux distribution that need modification to the "${TVM_HOME}"/python/setup.py. Please refer to the \"running_on_zynq.md\" document for more instruction." 
fi

