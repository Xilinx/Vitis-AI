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

vai_add_library(SRCS src/extra_ops.cpp PUBLIC_REQUIRE xir::xir unilog::unilog)
target_link_libraries(${COMPONENT_NAME} PRIVATE gcc_atomic)
if(BUILD_PYTHON)
  vai_add_pybind11_module(xir_extra_ops src/extra_ops_py.cpp)
  target_link_libraries(xir_extra_ops PRIVATE ${COMPONENT_NAME})
endif(BUILD_PYTHON)
