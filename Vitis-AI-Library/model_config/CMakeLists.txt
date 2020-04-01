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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include(XilinxCommon)
include(XilinxVersion)


include_directories(${Protobuf_INCLUDE_DIRS})

add_custom_command(
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/proto/dpu_model_param.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/proto/dpu_model_param.pb.h
    DEPENDS   ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai/proto/dpu_model_param.proto
    COMMAND  protobuf::protoc
    ARGS --cpp_out  ${CMAKE_CURRENT_BINARY_DIR}
    --proto_path ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai/proto/dpu_model_param.proto
    COMMENT "Running C++ protocol buffer compiler on proto dpu_model_param"
    VERBATIM )

add_library(${COMPONENT_NAME} SHARED
	${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/proto/dpu_model_param.pb.cc
	${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/proto/dpu_model_param.pb.h
    src/config.cpp
  )
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
target_link_libraries(${COMPONENT_NAME} glog protobuf::libprotobuf)
set_target_properties(${COMPONENT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
  LIBRARY_OUTPUT_NAME  ${PROJECT_NAME}-${COMPONENT_NAME})


install(FILES
  include/vitis/ai/proto/config.hpp
  include/vitis/ai/proto/dpu_model_param.proto
  ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/proto/dpu_model_param.pb.h
  DESTINATION include/vitis/ai/proto)

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  RUNTIME DESTINATION ${INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${INSTALL_LIB_DIR})

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${INSTALL_CMAKE_DIR})
