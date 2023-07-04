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

vai_add_library(
  SRCS ${LIST_PBSRC_CC}
       src/fusion_cnn.cpp
       src/fusion_cnn_imp.hpp
       src/fusion_cnn_imp.cpp
  PUBLIC_REQUIRE ${PROJECT_NAME}::xnnpp ${PROJECT_NAME}::dpu_task xir::xir)

target_include_directories(${COMPONENT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${COMPONENT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

vai_add_test(test_fusion_cnn)
vai_add_test(test_fusion_cnn_batch)
