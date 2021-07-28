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
cmake_minimum_required(VERSION 3.9)
project(plugin-samples LANGUAGES C CXX)

include(${CMAKE_SOURCE_DIR}/cmake/VitisCommon.cmake)

find_package(unilog REQUIRED)
find_package(xir REQUIRED)
find_package(target-factory REQUIRED)
find_package(xcompiler COMPONENTS xcompiler-core REQUIRED)

add_subdirectory(samples)
