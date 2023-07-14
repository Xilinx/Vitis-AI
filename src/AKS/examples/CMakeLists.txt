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

project(examples VERSION 2.5.0)

# Set source files of examples (non-video)
set (EXAMPLES_IMAGE
  tf_resnet_v1_50.cpp
  cf_resnet50.cpp
  cf_densebox_320_320.cpp
)

# Set source files of examples (video)
set (EXAMPLES_VIDEO
  ""
)

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
message(STATUS "AKS Includes: ${aks_INCLUDE_DIRS}")

execute_process(COMMAND uname -m OUTPUT_VARIABLE arch)
find_package(Threads REQUIRED)
if(${arch} MATCHES ".*x86.*" AND NOT DEFINED ENV{SDKTARGETSYSROOT})
  find_package(Boost 1.65.1 EXACT REQUIRED COMPONENTS system filesystem)
else()
  find_package(Boost 1.65.1 REQUIRED COMPONENTS system filesystem)
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc video videoio imgcodecs)

# Set output directories for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Compile Sources
foreach( sourcefile ${EXAMPLES_IMAGE} )
  # Generate executable name
  string( REPLACE ".cpp" ".exe" exename ${sourcefile} )

  # Set target
  add_executable( ${exename} ${sourcefile} )

  # Set include dirs
  target_include_directories (${exename}
    PRIVATE ${aks_INCLUDE_DIRS}
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
  )

  # Set libraries to be linked
  target_link_libraries (${exename}
    PRIVATE ${aks_LIBRARIES}
    PRIVATE ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
    PRIVATE ${CMAKE_DL_LIBS}
    PRIVATE ${CMAKE_THREAD_LIBS_INIT}
    PRIVATE vart-runner
    PRIVATE xir unilog
    PRIVATE opencv_core opencv_imgproc opencv_imgcodecs
  )
  # set (CMAKE_DEBUG_TARGET_PROPERTIES INCLUDE_DIRECTORIES)
endforeach()

# Compile Sources
foreach( sourcefile ${EXAMPLES_VIDEO} )
  # Generate executable name
  string( REPLACE ".cpp" ".exe" exename ${sourcefile} )

  # Set target
  add_executable( ${exename} ${sourcefile} )

  # Set include dirs
  target_include_directories (${exename}
    PRIVATE ${aks_INCLUDE_DIRS}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
  )

  # Set libraries to be linked
  target_link_libraries (${exename}
    PRIVATE ${aks_LIBRARIES}
    PRIVATE ${CMAKE_DL_LIBS}
    PRIVATE ${CMAKE_THREAD_LIBS_INIT}
    PRIVATE opencv_core opencv_imgproc opencv_video opencv_videoio
    PRIVATE vart-runner
    PRIVATE xir unilog
  )
  # set (CMAKE_DEBUG_TARGET_PROPERTIES INCLUDE_DIRECTORIES)
endforeach()
