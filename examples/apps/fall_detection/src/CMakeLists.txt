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

project(fall_detection VERSION 1.3.0)

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

find_package(Threads REQUIRED)
find_package(Boost 1.65.1 EXACT REQUIRED COMPONENTS system filesystem)
find_package(OpenCV REQUIRED core imgcodecs videoio)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set (SRC
  main.cpp
)

foreach( sourcefile ${SRC} )
  string( REPLACE ".cpp" ".exe" exename ${sourcefile} )
  add_executable( ${exename} ${sourcefile} )
  target_include_directories (${exename}
    PRIVATE ${aks_INCLUDE_DIRS}
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries (${exename}
    PRIVATE ${aks_LIBRARIES}
    PRIVATE ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY}
    PRIVATE ${CMAKE_DL_LIBS}
    PRIVATE ${CMAKE_THREAD_LIBS_INIT}
    PRIVATE opencv_core opencv_imgcodecs opencv_videoio
  )
endforeach()
