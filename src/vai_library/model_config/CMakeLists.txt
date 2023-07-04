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

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/proto/dpu_model_param.pb.cc
         ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/proto/dpu_model_param.pb.h
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai/proto/dpu_model_param.proto
  COMMAND
    protobuf::protoc ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai/proto/dpu_model_param.proto
  COMMENT "Running C++ protocol buffer compiler on proto dpu_model_param"
  VERBATIM)

vai_add_library(
  SRCS
  ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/proto/dpu_model_param.pb.cc
  ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/proto/dpu_model_param.pb.h
  src/config.cpp #
  src/error_code.cpp
  PUBLIC_REQUIRE
  protobuf::libprotobuf
  glog::glog unilog::unilog)

target_include_directories(
  ${COMPONENT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/proto/dpu_model_param.pb.h
              include/vitis/ai/proto/dpu_model_param.proto
        DESTINATION include/vitis/ai/proto)
# include/vitis/ai/proto/config.hpp include/vitis/ai/proto/dpu_model_param.proto

if(BUILD_PYTHON)
  add_custom_target(
    py_dpu_model_param_proto
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai/proto/dpu_model_param.proto
    COMMAND
      protobuf::protoc --python_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai/proto/dpu_model_param.proto
    COMMENT
      "Running PYTHON protocol buffer compiler on proto dpu_model_param.proto"
    VERBATIM)
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/proto/dpu_model_param_pb2.py
    DESTINATION ${PYTHON_SITE_PACKAGES}/vitis/ai/proto/)
endif(BUILD_PYTHON)
