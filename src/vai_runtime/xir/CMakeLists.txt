#
# Copyright 2019 Xilinx Inc.
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

cmake_minimum_required(VERSION 3.9)
project(
  xir
  VERSION 3.0.0
  LANGUAGES C CXX)

include(${PROJECT_SOURCE_DIR}/cmake/VitisCommon.cmake)

option(BUILD_CONTRIB "build contrib lib" OFF)
option(BUILD_TEST "build test bin" OFF)
option(BUILD_DOC "build documentation" OFF)
option(BUILD_PYTHON "build python interface" OFF)
option(INSTALL_HOME "install python lib in cmake install path" OFF)
option(INSTALL_USER "install python lib in user space" OFF)

# find_package(glog REQUIRED)
if(NOT TARGET unilog::unilog)
  find_package(unilog REQUIRED)
endif()

find_package(Boost REQUIRED)
if(BUILD_SHARED_LIBS)
    set(Protobuf_USE_STATIC_LIBS OFF)
else(BUILD_SHARED_LIBS)
    set(Protobuf_USE_STATIC_LIBS ON)
endif(BUILD_SHARED_LIBS)
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory(src)
add_subdirectory(tools)

file(RELATIVE_PATH REL_INCLUDE_DIR
     ${CMAKE_INSTALL_PREFIX}/share/cmake/${PROJECT_NAME}
     ${CMAKE_INSTALL_PREFIX}/include)
set(CONF_INCLUDE_DIRS "\${${PROJECT_NAME}_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake/config.cmake.in
               "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake" @ONLY)

write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
              "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
        DESTINATION share/cmake/${PROJECT_NAME})

if(BUILD_TEST)
  add_subdirectory(test)
endif()

if(BUILD_DOC)
  add_subdirectory(doc/doxygen)
endif(BUILD_DOC)

set(CPACK_PACKAGE_NAME "libxir")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Xilinx Intermediate Representation for deep learning algorithms")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_README "${PROJECT_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/Copyright.txt")
set(CPACK_PACKAGE_VENDOR "Humanity")
set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_CONTACT "Jian Weng<jianweng@xilinx.com>")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

set(CPACK_DEB_COMPONENT_INSTALL OFF)
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")
set(CPACK_DEBIAN_COMPRESSION_TYPE "gzip")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libunilog (>=1.2.0)")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
if(DEFINED ENV{BUILD_NUMBER})
  set(CPACK_DEBIAN_PACKAGE_RELEASE "r$ENV{BUILD_NUMBER}")
endif()
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
else()
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
endif()

set(CPACK_RPM_COMPONENT_INSTALL OFF)
set(CPACK_RPM_PACKAGE_LICENSE "Apache")
set(CPACK_RPM_PACKAGE_GROUP "Development/Tools")
set(CPACK_RPM_COMPRESSION_TYPE "xz")
set(CPACK_RPM_PACKAGE_REQUIRES "libunilog >= 1.2.0")
set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
if(DEFINED ENV{BUILD_NUMBER})
  set(CPACK_RPM_PACKAGE_RELEASE "r$ENV{BUILD_NUMBER}")
endif()
set(CPACK_RPM_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortexa72-cortexa53")
  set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
endif()

include(CPack)
