#
# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
cmake_minimum_required(VERSION 3.12)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/vitis_ai_library_common.cmake)
project(
  hello_vitis_ai_lib
  VERSION 1.0.0
  LANGUAGES C CXX)
# find_package(vitis_ai_library REQUIRED) find_package(OpenCV REQUIRED core
# videoio) already included by vitis-ai-library
find_package(glog REQUIRED)
message("OpenCV_LIBS is ${OpenCV_LIBS}")
vai_add_library(
  SRCS src/message1.cpp src/message2.cc
  PUBLIC_HEADER include/fancy/1/message1.hpp include/fancy/message2.hpp
  PRIVATE_REQUIRE glog::glog)
add_executable(showargs test/showargs.cpp xx.cpp)
target_link_libraries(showargs ${PROJECT_NAME}::${COMPONENT_NAME})
install(TARGETS showargs DESTINATION bin)
