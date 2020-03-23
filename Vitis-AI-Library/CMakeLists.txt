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
cmake_minimum_required(VERSION 3.5)
project(vitis_ai_library VERSION 1.1.0 LANGUAGES C CXX)
include(${CMAKE_SOURCE_DIR}/cmake/XilinxCommon.cmake)

## ugly solution for search XRT library, xrt is built without proper rpath setting
link_directories("/opt/xilinx/xrt/lib")
## options
option(DISABLE_ALL "disable all libraries" OFF)
option(ENABLE_CLASSIFICATION "enable classification" OFF)
option(ENABLE_TFSSD "enable tfssd" OFF)
option(ENABLE_FACEDETECT "enable face detection" OFF)
option(ENABLE_LANEDETECT "enable lane detection" OFF)
option(ENABLE_YOLOV2 "enable Yolov2" OFF)
option(ENABLE_YOLOV3 "enable Yolov3" OFF)
option(ENABLE_LANDMARK "enable landmark face-5pt" OFF)
option(ENABLE_SEGMENTATION "enable segmentation" OFF)
option(ENABLE_REFINEDET "enable refinedet" OFF)
option(ENABLE_OPENPOSE "enable openpose" OFF)
option(ENABLE_POSEDETECT "enable pose detect" OFF)
option(ENABLE_REID "enable reid" OFF)
option(ENABLE_MULTITASK "enable multitask" OFF)
option(ENABLE_OVERVIEW "enable overview" OFF)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_LIST_DIR}/cmake")
set(INSTALL_LIB_DIR lib)
set(INSTALL_BIN_DIR bin)
set(INSTALL_INCLUDE_DIR include)
set(INSTALL_CMAKE_DIR share/cmake/${PROJECT_NAME})

include_directories(${CMAKE_BINARY_DIR}/model_config)
include_directories("${CMAKE_INSTALL_PREFIX}/include")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

find_package(vart COMPONENTS util REQUIRED)
find_package(Protobuf 3.4.0 REQUIRED)

add_subdirectory(model_config)

add_subdirectory(math)
include_directories(math/include)
link_directories(${CMAKE_BINARY_DIR}/math)

add_subdirectory(dpu_task)
include_directories(dpu_task/include)

add_subdirectory(usefultools)

add_subdirectory(xnnpp)
include_directories(xnnpp/include)

if(ENABLE_CLASSIFICATION OR (NOT DISABLE_ALL))
  add_subdirectory(classification)
  include_directories(classification/include)
endif()

if(ENABLE_TFSSD OR (NOT DISABLE_ALL))
  add_subdirectory(tfssd)
  include_directories(tfssd/include)
endif()
if(ENABLE_FACEDETECT OR (NOT DISABLE_ALL))
  add_subdirectory(facedetect)
  include_directories(facedetect/include)
endif()
if(ENABLE_LANEDETECT OR (NOT DISABLE_ALL))
  add_subdirectory(lanedetect)
  include_directories(lanedetect/include)
endif()

if(ENABLE_YOLOV2 OR (NOT DISABLE_ALL))
  add_subdirectory(yolov2)
  include_directories(yolov2/include)
endif()

if(ENABLE_YOLOV3 OR (NOT DISABLE_ALL))
  add_subdirectory(yolov3)
  include_directories(yolov3/include)
endif()
if(ENABLE_LANDMARK OR (NOT DISABLE_ALL))
  add_subdirectory(facelandmark)
  include_directories(facelandmark/include)
endif()
if(ENABLE_SSD OR (NOT DISABLE_ALL))
  add_subdirectory(ssd)
  include_directories(ssd/include)
endif()
if(ENABLE_SEGMENTATION OR (NOT DISABLE_ALL))
  add_subdirectory(segmentation)
  include_directories(segmentation/include)
endif()

if(ENABLE_REFINEDET OR (NOT DISABLE_ALL))
  add_subdirectory(refinedet)
  include_directories(refinedet/include)
endif()

if(ENABLE_OPENPOSE OR (NOT DISABLE_ALL))
  add_subdirectory(openpose)
  include_directories(openpose/include)
endif()

if(ENABLE_POSEDETECT OR (NOT DISABLE_ALL))
  add_subdirectory(posedetect)
  include_directories(posedetect/include)
endif()

if(ENABLE_REID OR (NOT DISABLE_ALL))
  add_subdirectory(reid)
  include_directories(reid/include)
endif()

if(ENABLE_MULTITASK OR (NOT DISABLE_ALL))
  add_subdirectory(multitask)
  include_directories(multitask/include)
endif()

add_subdirectory(benchmark)
include_directories(benchmark/include)

if(ENABLE_OVERVIEW)
  add_subdirectory(overview)
endif()

set(CPACK_PACKAGE_NAME "libvitis_ai_library")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A runtime runner for vitis")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/Copyright.txt")
set(CPACK_PACKAGE_VENDOR "Humanity")
set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_CONTACT "Hui Zhang<huizhang@xilinx.com>")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}.${CMAKE_PROJECT_VERSION_PATCH}")
if(DEFINED ENV{BUILD_NUMBER})
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-build$ENV{BUILD_NUMBER}")
endif()

set(CPACK_DEB_COMPONENT_INSTALL False)
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")
set(CPACK_DEBIAN_COMPRESSION_TYPE "gzip")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libvart (>=0.0.1)")

set(CPACK_RPM_COMPONENT_INSTALL False)
set(CPACK_RPM_PACKAGE_LICENSE "Apache")
set(CPACK_RPM_PACKAGE_GROUP "Development/Tools")
set(CPACK_RPM_COMPRESSION_TYPE "xz")
set(CPACK_RPM_PACKAGE_REQUIRES "libvart >= 0.0.1")

include(CPack)
