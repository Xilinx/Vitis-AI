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

if(CPACK_DEBIAN_PACKAGE_ARCHITECTURE STREQUAL "arm64")
  add_compile_options(-Wno-error=attributes -Wno-strict-aliasing)
endif()

vai_add_library(
  SRCS ${LIST_PBSRC_CC}
       src/clocs.cpp
       src/clocs_imp.hpp
       src/clocs_imp.cpp
       src/clocs_pointpillars.hpp
       src/clocs_pointpillars.cpp
       src/clocs_pointpillars_imp.hpp
       src/clocs_pointpillars_imp.cpp
       src/voxelizer.hpp
       src/voxelizer.cpp
       src/scatter.hpp
       src/scatter.cpp
       src/anchor.hpp
       src/anchor.cpp
       src/utils.hpp
       src/utils.cpp
  PUBLIC_REQUIRE ${PROJECT_NAME}::yolovx ${PROJECT_NAME}::fusion_cnn ${PROJECT_NAME}::xnnpp ${PROJECT_NAME}::dpu_task xir::xir)

target_include_directories(${COMPONENT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${COMPONENT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

vai_add_test(test_clocs)
vai_add_test(test_clocs_batch)
#vai_add_test(test_clocs_with_2dresult)
#vai_add_test(test_clocs_with_2dresult_2)
vai_add_test(test_clocs_pointpillars)
vai_add_test(test_clocs_select_result)
#vai_add_test(test_clocs_pointpillars_batch)
#vai_add_test(test_clocs_pointpillars_scatter)
