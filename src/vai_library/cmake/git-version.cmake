# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Distributed under the Apache License, Version 2.0 (the "License"); you may not
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
# in case Git is not available, we default to "unknown"
set(GIT_VERSION "unknown")

find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --dirty --long
    OUTPUT_VARIABLE GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()

if ("${GIT_VERSION_SHORT}" STREQUAL "")
  execute_process(
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND git rev-parse HEAD
    OUTPUT_VARIABLE GIT_VERSION_SHORT)
endif()
string(STRIP "${GIT_VERSION_SHORT}" GIT_VERSION_SHORT)

message(STATUS "Git hash is ${GIT_VERSION}")
include(${CURRENT_BINARY_DIR}/xilinx_version.c.kv)
string(TIMESTAMP BUILT_TIME "%Y-%m-%d %H:%M:%S [UTC]" UTC)
configure_file(${CMAKE_CURRENT_LIST_DIR}/xilinx_version_2.c.in
               ${CURRENT_BINARY_DIR}/xilinx_version.c @ONLY)
