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

OPENCV_VERSION=4.5.5
wget https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz
tar xf ${OPENCV_VERSION}.tar.gz -C /tmp
cd /tmp
mkdir -p build
cd build
cmake  ../opencv-${OPENCV_VERSION} -D CMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}/opencv-${OPENCV_VERSION} -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF -DBUILD_TIFF=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DWITH_PROTOBUF=OFF
make -j $(nproc)
sudo make install 