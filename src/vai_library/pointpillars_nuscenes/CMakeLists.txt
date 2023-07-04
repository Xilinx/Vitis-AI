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

if(CPACK_DEBIAN_PACKAGE_ARCHITECTURE STREQUAL "arm64")
  add_compile_options(-Wno-error=attributes -Wno-strict-aliasing)
endif()

find_package(Eigen3 REQUIRED)

#if(HAVE_NEON)
#  add_definitions(-DENABLE_NEON)
#  list(APPEND LIBRARY_SOURCE src/softmax_table.hpp src/softmax_table.cpp)
#endif()

vai_add_library(
  SRCS ${LIST_PBSRC_CC}
       src/pointpillars_nuscenes.cpp
       src/pointpillars_nuscenes_imp.hpp
       src/pointpillars_nuscenes_imp.cpp
       src/voxelize.hpp
       src/voxelize.cpp
       src/scatter.hpp
       src/scatter.cpp
       src/utils.hpp
       src/utils.cpp
       src/multi_frame_fusion.hpp
       src/multi_frame_fusion.cpp
  PUBLIC_REQUIRE ${PROJECT_NAME}::xnnpp ${PROJECT_NAME}::dpu_task xir::xir)

target_include_directories(${COMPONENT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${COMPONENT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

vai_add_test(test_pointpillars_nus)
vai_add_test(test_pointpillars_nus_batch)
#vai_add_test(test_pointpillars_nus_voxelize)
vai_add_test(test_pointpillars_nus_scatter)
#vai_add_test(test_pointpillars_nus_scatter2)
vai_add_test(test_pointpillars_nus_fusion)
#vai_add_test(test_pointpillars_accuracy)
