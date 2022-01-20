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

cmake_minimum_required(VERSION 3.12)
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0 -fno-inline")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

project(OpticalFlowOpenCV VERSION 1.3.0)

if (NOT "${AKS_INSTALL_PREFIX}" STREQUAL "")
  message(STATUS "AKS Install Prefix: ${AKS_INSTALL_PREFIX}")
  find_package(aks REQUIRED
    PATHS ${AKS_INSTALL_PREFIX}
    NO_DEFAULT_PATH
  )
else()
  find_package(aks REQUIRED
  )
endif()
message(STATUS "AKS Includes: ${aks_INCLUDE_DIRS}")

find_package(OpenCV REQUIRED COMPONENTS core imgproc video)

add_library (${PROJECT_NAME} SHARED
  AksOpticalFlowOpenCV.cpp
)

set_target_properties (${PROJECT_NAME} PROPERTIES
  VERSION   "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_include_directories (${PROJECT_NAME}
  PRIVATE ${aks_INCLUDE_DIRS}
  PRIVATE ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries (${PROJECT_NAME}
 PRIVATE opencv_core opencv_imgproc opencv_video
 PRIVATE ${aks_LIBRARIES}
)

