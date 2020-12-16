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

project(examples VERSION 1.3.0)

if (DPU STREQUAL "dpucadx8g")
# Set source files of examples (non-video)
set (EXAMPLES_IMAGE
  facedetect.cpp
  googlenet.cpp
  googlenet_pp_accel.cpp
  googlenet_resnet50.cpp
  googlenet_tinyyolov3.cpp
  inception_v1_tf.cpp
  resnet50.cpp
  stdyolov2.cpp
  tinyyolov3.cpp
)
# Set source files of examples (video)
set (EXAMPLES_VIDEO
  tinyyolov3_video.cpp
)
elseif (DPU STREQUAL "dpucahx8h")
# Set source files of examples (non-video)
set (EXAMPLES_IMAGE
  resnet50_u50.cpp
)
# Set source files of examples (video)
set (EXAMPLES_VIDEO
  ""
)
elseif (DPU STREQUAL "dpuczdx8g")
# Set source files of examples (non-video)
set (EXAMPLES_IMAGE
  resnet50_edge.cpp
  )
# Set source files of examples (video)
set (EXAMPLES_VIDEO
  ""
)
elseif (DPU STREQUAL "dpucadf8h")
# Set source files of examples (non-video)
set (EXAMPLES_IMAGE
  resnet50_dpucadf8h.cpp
  )
# Set source files of examples (video)
set (EXAMPLES_VIDEO
  ""
)
else()
message (FATAL_ERROR "
  Nothing to build.
  Make sure to specify target DPU with \'--dpu\' option."
  )
endif()

# Find Packages
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

execute_process(COMMAND uname -m OUTPUT_VARIABLE arch)
find_package(Threads REQUIRED)
if(${arch} MATCHES ".*x86.*")
  find_package(Boost 1.65.1 EXACT REQUIRED COMPONENTS system filesystem)
else()
  find_package(Boost 1.65.1 REQUIRED COMPONENTS system filesystem)
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc video videoio)

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
  )

  # Set libraries to be linked
  target_link_libraries (${exename}
    PRIVATE ${aks_LIBRARIES}
    PRIVATE ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
    PRIVATE ${CMAKE_DL_LIBS}
    PRIVATE ${CMAKE_THREAD_LIBS_INIT}
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
  )
  # set (CMAKE_DEBUG_TARGET_PROPERTIES INCLUDE_DIRECTORIES)
endforeach()
