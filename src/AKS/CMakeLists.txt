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
# Project Name and Version
project(aks VERSION 2.5.0 LANGUAGES C CXX)

include(${CMAKE_SOURCE_DIR}/cmake/VitisCommon.cmake)

# Project library
add_subdirectory(src)

# Enable/Disable tests
option(BUILD_TEST "Build Tests" OFF)
if(BUILD_TEST)
  add_subdirectory(tests)
endif()

# Project Packaging and cmake configs
file(RELATIVE_PATH REL_INCLUDE_DIR
  ${CMAKE_INSTALL_PREFIX}/share/cmake/${PROJECT_NAME}
  ${CMAKE_INSTALL_PREFIX}/include
)

set(CONF_INCLUDE_DIRS "\${${PROJECT_NAME}_CMAKE_DIR}/${REL_INCLUDE_DIR}")

configure_file(cmake/config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  @ONLY
)

write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  DESTINATION share/cmake/${PROJECT_NAME}
)

set(CMAKE_PROJECT_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CMAKE_PROJECT_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CMAKE_PROJECT_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set(CPACK_PACKAGE_NAME "aks")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AI Kernel Scheduler")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/Copyright.txt")
set(CPACK_PACKAGE_VENDOR "Xilinx")
set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_CONTACT "Vishal Jain<vishal.jain@xilinx.com>")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

# .deb package
set(CPACK_DEB_COMPONENT_INSTALL OFF)
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")
set(CPACK_DEBIAN_COMPRESSION_TYPE "gzip")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
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

# .rpm package
set(CPACK_RPM_COMPONENT_INSTALL OFF)
set(CPACK_RPM_PACKAGE_LICENSE "Apache")
set(CPACK_RPM_PACKAGE_GROUP "Development/Tools")
set(CPACK_RPM_COMPRESSION_TYPE "xz")
set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
if(DEFINED ENV{BUILD_NUMBER})
  set(CPACK_RPM_PACKAGE_RELEASE "r$ENV{BUILD_NUMBER}")
endif()
set(CPACK_RPM_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortexa72-cortexa53")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
endif()

include(CPack)
