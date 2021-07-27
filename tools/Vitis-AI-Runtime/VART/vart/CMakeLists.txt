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
project(vart VERSION 1.4.0 LANGUAGES C CXX)

link_directories("$ENV{CONDA_PREFIX}/lib")
include(${CMAKE_SOURCE_DIR}/cmake/VitisCommon.cmake)

option(ENABLE_CPU_RUNNER "enable cpu runner or not" OFF)
option(ENABLE_SIM_RUNNER "enable sim runner or not" OFF)
option(ENABLE_DPU_RUNNER "enable dpu runner or not" OFF)
option(ENABLE_XRNN_RUNNER "enable xrnn runner or not" OFF)
option(ENABLE_RNN_RUNNER "enable rnn runner or not" OFF)
option(BUILD_PYTHON "build python library" OFF)
option(BUILD_TEST "build test" OFF)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake")
find_package(Protobuf REQUIRED)
find_package(unilog REQUIRED)
find_package(xir REQUIRED)
find_package(target-factory REQUIRED)
find_package(Xrt MODULE)
find_package(hiredis QUIET)

set(PACKAGE_COMPONENTS util trace runner dummy-runner async-runner)
list(APPEND PACKAGE_COMPONENTS mem-manager)
if(ENABLE_DPU_RUNNER)
  # append xrt-device-handle anyway, even if xrt is not found, because
  # softmax controller is part if and vitis-ai-library needs such
  # components.
  list(APPEND PACKAGE_COMPONENTS xrt-device-handle)
  list(APPEND PACKAGE_COMPONENTS buffer-object dpu-controller dpu-runner softmax-runner softmax-runner-cpu)
endif()
if(ENABLE_SIM_RUNNER)
  list(APPEND PACKAGE_COMPONENTS sim-runner)
endif()
if(ENABLE_CPU_RUNNER)
  list(APPEND PACKAGE_COMPONENTS cpu-runner)
endif()
if(ENABLE_XRNN_RUNNER)
  list(APPEND PACKAGE_COMPONENTS xrt-device-handle)
  list(APPEND PACKAGE_COMPONENTS buffer-object xrnn-runner)
endif()
if(ENABLE_RNN_RUNNER)
  list(APPEND PACKAGE_COMPONENTS xrt-device-handle)
  list(APPEND PACKAGE_COMPONENTS buffer-object rnn-runner)
endif()

list(REMOVE_DUPLICATES PACKAGE_COMPONENTS)
foreach(COMPONENT ${PACKAGE_COMPONENTS})
  add_subdirectory(${COMPONENT})
endforeach()

file(RELATIVE_PATH REL_INCLUDE_DIR
  ${CMAKE_INSTALL_PREFIX}/share/cmake/${PROJECT_NAME}
  ${CMAKE_INSTALL_PREFIX}/include)
set(CONF_INCLUDE_DIRS "\${${PROJECT_NAME}_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake/config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  @ONLY)

write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  COMPONENT base
  DESTINATION share/cmake/${PROJECT_NAME})

set(CPACK_PACKAGE_NAME "libvart")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Vitis Ai RuntTime")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/Copyright.txt")
set(CPACK_PACKAGE_VENDOR "Humanity")
set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_CONTACT "Hui Zhang<huizhang@xilinx.com>")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")
set(CPACK_DEBIAN_COMPRESSION_TYPE "gzip")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libxir (>=1.2.0), libunilog (>=1.2.0), libtarget-factory (>=1.2.0)")
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

set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_PACKAGE_LICENSE "Apache")
set(CPACK_RPM_PACKAGE_GROUP "Development/Tools")
set(CPACK_RPM_COMPRESSION_TYPE "xz")
set(CPACK_RPM_PACKAGE_REQUIRES "libxir >= 1.2.0, libunilog >= 1.2.0, libtarget-factory >= 1.2.0")
set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
if(DEFINED ENV{BUILD_NUMBER})
  set(CPACK_RPM_PACKAGE_RELEASE "r$ENV{BUILD_NUMBER}")
endif()
set(CPACK_RPM_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortexa72-cortexa53")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
endif()


set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)

include(CPack)
cpack_add_component_group(all)
cpack_add_component(base GROUP all)
cpack_add_component(cpu GROUP all DEPENDS base)
cpack_add_component(dpu GROUP all DEPENDS base)
cpack_add_component(sim GROUP all DEPENDS base)
