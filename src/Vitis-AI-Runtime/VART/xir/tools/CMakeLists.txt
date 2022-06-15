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

add_executable(
  xir_util
  xir_util.hpp
  xir_util.cpp
  cmd_dump_reg.hpp
  cmd_dump_reg.cpp
  cmd_png.hpp
  cmd_png.cpp
  cmd_svg.hpp
  cmd_svg.cpp
  cmd_dump_txt.hpp
  cmd_dump_txt.cpp
  cmd_subgraph.hpp
  cmd_subgraph.cpp
  cmd_dump_code.hpp
  cmd_dump_code.cpp
  cmd_graph.hpp
  cmd_graph.cpp)
target_link_libraries(xir_util xir protobuf::libprotobuf unilog::unilog)
target_include_directories(xir_util
                           PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../src/xir)
set_target_properties(xir_util PROPERTIES OUTPUT_NAME xir)

install(TARGETS xir_util DESTINATION bin)
