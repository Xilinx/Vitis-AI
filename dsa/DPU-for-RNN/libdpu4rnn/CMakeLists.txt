#
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
#

cmake_minimum_required(VERSION 2.6)
project(dpu4rnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
 
LINK_DIRECTORIES(/usr/lib)
LINK_DIRECTORIES(/usr/local/lib)
LINK_DIRECTORIES(/opt/xilinx/xrt/lib)
link_directories(/opt/vitis_ai/conda/envs/rnn_tf_2.0/lib/python3.7/)

INCLUDE_DIRECTORIES(/usr/include)
INCLUDE_DIRECTORIES(/usr/local/include)
INCLUDE_DIRECTORIES(/opt/xilinx/xrt/include)
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(src)
include_directories(/opt/vitis_ai/conda/envs/rnn_tf_2.0/include/pybind11/)
include_directories(/opt/vitis_ai/conda/envs/rnn_tf_2.0/include/python3.7m/)

set(LIB_NAME ${PROJECT_NAME})
set(PY_LIB_NAME dpu4rnn_py)

set(DEPS_PUBLIC xir unilog vart-runner vart-xrnn-runner uuid util glog)
set(DEPS_PRIVATE  vart-xrt-device-handle vart-buffer-object)


aux_source_directory(src SOURCE_FILES)
add_library(${LIB_NAME} SHARED  ${SOURCE_FILES})
target_link_libraries(${LIB_NAME} vart-runner xir glog)

add_executable(test ${SOURCE_FILES} test.cpp)
target_link_libraries(test vart-runner xir glog)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

add_library(${PY_LIB_NAME} SHARED python/dpu4rnn_python.cpp ${SOURCE_FILES})
set_target_properties(${PY_LIB_NAME} PROPERTIES PREFIX "")
target_link_libraries(${PY_LIB_NAME} vart-runner xir glog)

