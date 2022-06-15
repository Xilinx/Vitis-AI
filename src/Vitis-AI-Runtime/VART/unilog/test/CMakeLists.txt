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

include_directories(${PROJECT_SOURCE_DIR}/include)

aux_source_directory(. TAR_FILE_LIST)
foreach(TAR_FILE ${TAR_FILE_LIST})
  get_filename_component(TAR_NAME ${TAR_FILE} NAME_WE)
  set(TAR_NAME ${TAR_NAME}.bin)

  add_executable(${TAR_NAME} ${TAR_FILE})
  target_link_Libraries(${TAR_NAME} ${PROJECT_NAME} unilog)
endforeach(TAR_FILE)
