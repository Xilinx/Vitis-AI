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

cmake_minimum_required(VERSION 3.12)
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0 -fno-inline")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

project(YoloPostProcessTVM VERSION 1.4.0)
find_package(xir REQUIRED)
find_package(unilog REQUIRED)
find_package(vart REQUIRED)

# add_library(dpuv1utils SHARED IMPORTED)

if (NOT "${AKS_INSTALL_PREFIX}" STREQUAL "")
  message(STATUS "AKS Install Prefix: ${AKS_INSTALL_PREFIX}")
  find_package(aks REQUIRED
    PATHS ${AKS_INSTALL_PREFIX}
    NO_DEFAULT_PATH
  )
  #set_target_properties(dpuv1utils PROPERTIES
  #  IMPORTED_LOCATION "$ENV{VAI_ALVEO_ROOT}/vai/dpuv1/utils/libdpuv1utils.so"
  #  INTERFACE_INCLUDE_DIRECTORIES "$ENV{VAI_ALVEO_ROOT}/vai/dpuv1/utils/postproc"
  #)
else()
  find_package(aks REQUIRED
  )
  #set_target_properties(dpuv1utils PROPERTIES
  #  IMPORTED_LOCATION "$ENV{CONDA_PREFIX}/lib/python3.6/site-packages/vai/dpuv1/utils/libdpuv1utils.so"
  #  INTERFACE_INCLUDE_DIRECTORIES "$ENV{CONDA_PREFIX}/lib/python3.6/site-packages/vai/dpuv1/utils/postproc/"
  #)
endif()
message(STATUS "AKS Includes: ${aks_INCLUDE_DIRS}")

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library (${PROJECT_NAME} SHARED
  AksYoloPostProcTVM.cpp
  yolo.cpp
)

set_target_properties (${PROJECT_NAME} PROPERTIES
  VERSION   "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_include_directories (${PROJECT_NAME}
  PRIVATE ${OpenCV_INCLUDE_DIRS}
  PRIVATE ${aks_INCLUDE_DIRS}
)

target_link_libraries (${PROJECT_NAME}
  #PRIVATE dpuv1utils
  PRIVATE opencv_core opencv_imgproc
  PRIVATE ${aks_LIBRARIES}
  PRIVATE vart-runner
  PRIVATE xir unilog
)