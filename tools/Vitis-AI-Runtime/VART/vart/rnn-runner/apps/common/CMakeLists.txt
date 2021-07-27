#
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
#

cmake_minimum_required(VERSION 2.6)
project(py_rnn_graph)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")

LINK_DIRECTORIES(/usr/lib)
LINK_DIRECTORIES(/usr/local/lib)
link_directories(/opt/vitis_ai/conda/envs/rnn_tf_2.0/lib/python3.7/)

INCLUDE_DIRECTORIES(/usr/include)
INCLUDE_DIRECTORIES(/usr/local/include)
include_directories(/opt/vitis_ai/conda/envs/rnn_tf_2.0/include/pybind11/)
include_directories(/opt/vitis_ai/conda/envs/rnn_tf_2.0/include/python3.7m/)

set(PY_LIB_NAME py_rnn_graph)

add_library(${PY_LIB_NAME} SHARED py_rnn_graph.cpp)
set_target_properties(${PY_LIB_NAME} PROPERTIES PREFIX "")
target_link_libraries(${PY_LIB_NAME} xir)
