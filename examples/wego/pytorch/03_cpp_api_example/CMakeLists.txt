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
project(demo VERSION 1.0.0 LANGUAGES C CXX)
set(OPENCV_VER 4.5.5)
find_package(wego_torch REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV PATHS $ENV{CONDA_PREFIX}/opencv-${OPENCV_VER} NO_DEFAULT_PATH REQUIRED)
find_package(Protobuf REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_INCLUDE_PATH})
include_directories(include)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${WEGO_TORCH_INCLUDE_DIRS})
include_directories(${WEGO_TORCH_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS} )

set(CMAKE_BUILD_TYPE Release)
set(WARNING_FLAG -Wall)

set(CMAKE_CXX_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
set(CMAKE_VERBOSE_MAKEFILEON ON)


if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
endif()

# ld option to warn if symbols are undefined
set(WEGO_TORCH_NO_UNDEFINED_SYMBOLS "-Wl,--no-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)
add_executable(demo example.cpp)

target_link_libraries(demo PRIVATE ${WEGO_TORCH_LIBRARIES} PUBLIC ${TORCH_LIBRARIES} PUBLIC ${OpenCV_LIBS})
target_link_options(demo PRIVATE "-Wl,--unresolved-symbols=ignore-in-shared-libs")