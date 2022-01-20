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

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src/xir
                    ${PROJECT_SOURCE_DIR}/src)

aux_source_directory(. SRC_FILS)

foreach(FIL ${SRC_FILS})
  get_filename_component(FIL_NAME ${FIL} NAME_WE)
  set(BIN ${FIL_NAME}.bin)

  add_executable(${BIN} ${FIL})
  target_link_libraries(${BIN} protobuf::libprotobuf unilog::unilog
                        ${PROJECT_NAME})
endforeach()
