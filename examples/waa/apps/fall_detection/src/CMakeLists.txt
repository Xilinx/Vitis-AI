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
set(CMAKE_CXX_STANDARD 17)

project(examples VERSION 1.4.0)

# Find Packages
find_package(xir REQUIRED)
find_package(unilog REQUIRED)
find_package(vart REQUIRED)

if (NOT "${AKS_INSTALL_PREFIX}" STREQUAL "")
  message(STATUS "AKS Install Prefix: ${AKS_INSTALL_PREFIX}")
  find_package(aks REQUIRED
    PATHS ${AKS_INSTALL_PREFIX}
    NO_DEFAULT_PATH
  )
else()
  find_package(aks REQUIRED)
endif()

set (INCLUDE_DIRS "include")

message(STATUS "Includes: ${INCLUDE_DIRS}")
message(STATUS "AKS Includes: ${aks_INCLUDE_DIRS}")
message(STATUS "AKS libraries: ${aks_LIBRARIES}")

execute_process(COMMAND uname -m OUTPUT_VARIABLE arch)
find_package(Threads REQUIRED)
if(${arch} MATCHES ".*x86.*" AND (NOT DEFINED ENV{OECORE_TARGET_ARCH}))
  find_package(Boost 1.65.1 EXACT REQUIRED COMPONENTS system filesystem)
else()
  find_package(Boost 1.65.1 REQUIRED COMPONENTS system filesystem)
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc video videoio imgcodecs)

# Set output directories for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Compile Sources

# Generate executable name
string( REPLACE ".cpp" ".exe" exename "main.cpp" )

# Set target
add_executable( ${exename} "main.cpp" "function_pool.cpp" )

# Set include dirs
target_include_directories (${exename}
  PRIVATE ${aks_INCLUDE_DIRS}
  PRIVATE ${Boost_INCLUDE_DIRS}
  PRIVATE ${OpenCV_INCLUDE_DIRS}
  PRIVATE ${INCLUDE_DIRS}
)

# Set libraries to be linked
target_link_libraries (${exename}
  PRIVATE ${aks_LIBRARIES}
  PRIVATE ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
  PRIVATE ${CMAKE_DL_LIBS}
  PRIVATE ${CMAKE_THREAD_LIBS_INIT}
  PRIVATE "/opt/xilinx/xrt/lib"
  PRIVATE OpenCL
  PRIVATE vart-runner
  PRIVATE xir unilog 
  PRIVATE opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_video
)
# set (CMAKE_DEBUG_TARGET_PROPERTIES INCLUDE_DIRECTORIES)
