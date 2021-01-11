# Copyright 2019 Xilinx Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND date +%F-%T
  OUTPUT_VARIABLE BUILD_DATE)
string(STRIP "${BUILD_DATE}" BUILD_DATE)
if ("${GIT_VERSION}" STREQUAL "")
  execute_process(
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND git rev-parse HEAD
    OUTPUT_VARIABLE GIT_VERSION)
endif()
string(STRIP "${GIT_VERSION}" GIT_VERSION)
configure_file(${CMAKE_CURRENT_LIST_DIR}/vitis_version.c.in ${CMAKE_CURRENT_BINARY_DIR}/version.c)
