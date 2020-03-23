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
get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SYSROOT}/usr/share/cmake/")
include(XilinxCommon)
include(XilinxVersion)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

install(FILES
  include/vitis/ai/benchmark.hpp
  include/vitis/ai/image_list.hpp
  include/vitis/ai/stat_samples.hpp
  DESTINATION include/vitis/ai
  )

install(FILES
  include/vitis/ai/demo.hpp
  include/vitis/ai/dpdrm.hpp
  DESTINATION include/vitis/ai
  )
