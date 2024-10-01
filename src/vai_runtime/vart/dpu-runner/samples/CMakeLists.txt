#
# Copyright 2022-2023 Advanced Micro Devices Inc.
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

project(vart_samples)

find_package(OpenCV REQUIRED)
find_package(unilog REQUIRED)
find_package(xir REQUIRED)
find_package(
  vart
  COMPONENTS util
             xrt-device-handle
             buffer-object
             dpu-controller
             runner
             dpu-runner
             runner-assistant
             mem-manager
  REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/my_find_xrt.cmake)
include_directories(common)

function(vart_add_sample name)
  set(options "")
  set(oneValueArgs DESTINATION)
  set(multiValueArgs REQUIRE SRCS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  add_executable(${name} ${ARG_SRCS})
  target_link_libraries(
  ${name} unilog::unilog xir::xir vart::runner ${OpenCV_LIBS} pthread)

  if(ARG_REQUIRE)
    target_link_libraries(${name} ${ARG_REQUIRE})
  endif(ARG_REQUIRE)
endfunction(vart_add_sample)

vart_add_sample(
  adas_detection
  SRCS adas_detection/src/main.cc
       common/common.cpp)

vart_add_sample(
  pose_detection
  SRCS pose_detection/src/main.cpp
       pose_detection/src/ssd.cpp
       pose_detection/src/pose.cpp
       common/common.cpp)

vart_add_sample(
  resnet50
  SRCS resnet50/src/main.cc
       common/common.cpp)

vart_add_sample(
  resnet50_ext
  SRCS resnet50_ext/resnet50.cpp
  REQUIRE vart::dpu-controller vart::xrt-device-handle vart::buffer-object
          vart::util vart::runner-assistant)

vart_add_sample(
  resnet50_pt
  SRCS resnet50_pt/resnet50_pt.cpp
  REQUIRE vart::dpu-controller vart::xrt-device-handle vart::buffer-object
          vart::util vart::runner-assistant)

vart_add_sample(
  resnet50_zero_copy
  SRCS resnet50_zero_copy/resnet50.cpp
  REQUIRE XRT::xrt_core XRT::xrt_coreutil vart::dpu-controller vart::xrt-device-handle
          vart::buffer-object vart::util vart::runner-assistant)

vart_add_sample(
  segmentation
  SRCS segmentation/src/main.cc
       common/common.cpp)

vart_add_sample(
  squeezenet_pytorch
  SRCS squeezenet_pytorch/src/main.cc
       common/common.cpp)

vart_add_sample(
  video_analysis
  SRCS video_analysis/src/main.cc
       common/common.cpp)
