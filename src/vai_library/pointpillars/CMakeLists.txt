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

set(PBF_NAME2
    activations
    anchors
    box_coder
    input_reader
    layers
    losses
    model
    optimizer
    pipeline
    preprocess
    sampler
    second
    similarity
    target
    train
    voxel_generator)

foreach(PBF ${PBF_NAME2})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/second/protos/${PBF}.pb.cc
           ${CMAKE_CURRENT_BINARY_DIR}/second/protos/${PBF}.pb.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/second/protos/${PBF}.proto
    COMMAND
      protobuf::protoc ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/include/second/protos/${PBF}.proto
    COMMENT "Running C++ protocol buffer compiler on proto ${PBF}"
    VERBATIM)
  list(APPEND LIST_PBSRC_CC
       ${CMAKE_CURRENT_BINARY_DIR}/second/protos/${PBF}.pb.cc)
  list(APPEND LIST_PBSRC_H2
       ${CMAKE_CURRENT_BINARY_DIR}/second/protos/${PBF}.pb.h)
endforeach()

if(CPACK_DEBIAN_PACKAGE_ARCHITECTURE STREQUAL "arm64")
  add_compile_options(-Wno-error=attributes -Wno-strict-aliasing)
endif()

vai_add_library(
  SRCS ${LIST_PBSRC_CC}
       src/helper.cpp
       src/pointpillars.cpp
       src/pointpillars_imp.cpp
       src/preprocess.cpp
       src/postprocess/anchor.cpp
       src/postprocess/pointpillars_post.cpp
       src/postprocess/parse_display_result.cpp
  PUBLIC_REQUIRE ${PROJECT_NAME}::dpu_task Eigen3::Eigen)

target_include_directories(${COMPONENT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${COMPONENT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

vai_add_test(test_pointpillars)
vai_add_test(test_pointpillars_ptr)
vai_add_test(test_pointpillars_accuracy)
vai_add_test(test_pointpillars_batch)
