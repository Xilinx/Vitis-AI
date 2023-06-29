#
# Copyright 2022-2023 Advanced Micro Devices Inc.
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

vai_add_library(
  SRCS include/xir/cxir.h
       graph.cpp
       attrs2.cpp
       tensors.cpp
       subgraph.cpp
       op.cpp
       opdef.cpp
  PUBLIC_REQUIRE xir::xir glog::glog)

target_include_directories(
  cxir PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/headers/include
               ${CMAKE_CURRENT_SOURCE_DIR}/headers/src)
