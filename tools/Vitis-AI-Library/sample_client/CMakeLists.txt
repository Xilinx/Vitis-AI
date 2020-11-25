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
cmake_minimum_required(VERSION 3.12)
project(hello_vitis_ai_lib VERSION 1.0.0 LANGUAGES C CXX)
find_package(vitis_ai_library REQUIRED)
# find_package(OpenCV REQUIRED core videoio) already included by vitis-ai-library
message("OpenCV_LIBS is ${OpenCV_LIBS}")
add_executable(showargs showargs.cpp)
install(TARGETS showargs DESTINATION bin)
