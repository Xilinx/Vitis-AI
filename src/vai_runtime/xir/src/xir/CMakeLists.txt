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

include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/VitisVersion.cmake")

# get the git repo branch and commit id and write them into config.hpp
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE PROJECT_GIT_BRANCH_NAME
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE PROJECT_GIT_COMMIT_ID
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION
                                            VERSION_LESS 6.0)
  set(CMAKE_CXX_COMPILER_VERSION_LESS_THAN_6_0
      true
      CACHE BOOL "Use GNU C++ compiler and version is less than 6.0")
endif()
configure_file(config/config.hpp.in config.hpp @ONLY)

# run protoc
set(protobuf_MODULE_COMPATIBLE
    ON
    CACHE BOOL "")
file(GLOB_RECURSE PROTO_FILES proto/*.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS EXPORT_MACRO XIR_DLLESPEC
                                                         ${PROTO_FILES})
if(MSVC)
    if(BUILD_SHARED_LIBS)
      set_source_files_properties(
        ${PROTO_SRCS} PROPERTIES COMPILE_DEFINITIONS
                                 "XIR_DLLESPEC=__declspec(dllexport)")
    else(BUILD_SHARED_LIBS)
        set_source_files_properties(
            ${PROTO_SRCS} PROPERTIES COMPILE_DEFINITIONS
                                     "XIR_DLLESPEC=")
    endif(BUILD_SHARED_LIBS)
else(MSVC)
  set_source_files_properties(
    ${PROTO_SRCS}
    PROPERTIES COMPILE_DEFINITIONS
               "XIR_DLLESPEC=__attribute__((visibility(\"default\")))")
  set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_FLAGS
                                                       -Wno-unused-variable)
endif(MSVC)
aux_source_directory(util/ CPP_SRCS)
aux_source_directory(attrs/ CPP_SRCS)
aux_source_directory(op/ CPP_SRCS)
aux_source_directory(graph/ CPP_SRCS)
aux_source_directory(tensor/ CPP_SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rd-party/hash-library
                     CPP_SRCS)

add_library(${PROJECT_NAME}  ${PROTO_SRCS} ${CPP_SRCS}
   ${PROJECT_SOURCE_DIR}/include//xir/XirExport.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/op/op.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/op/op_def.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/attrs/attrs.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/attrs/attr_def.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/attrs/attr_expander.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/util/any.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/util/tool_function.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/util/data_type.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/graph/graph.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/graph/graph_template.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/graph/subgraph.hpp
  ${PROJECT_SOURCE_DIR}/include/xir/tensor/tensor.hpp
                             ${CMAKE_CURRENT_BINARY_DIR}/version.c)
set_target_properties(
  ${PROJECT_NAME} PROPERTIES VERSION "${PROJECT_VERSION}"
                             SOVERSION "${PROJECT_VERSION_MAJOR}")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/util ${Boost_INCLUDE_DIRS}
          ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
if(BUILD_SHARED_LIBS)
    target_link_libraries(
      xir
      PRIVATE protobuf::libprotobuf Threads::Threads ${CMAKE_DL_LIBS}
      PUBLIC unilog::unilog)
    if(NOT MSVC)
      if(Protobuf_USE_STATIC_LIBS)
        target_link_options(xir PRIVATE -Wl,--exclude-libs,ALL)
      endif(Protobuf_USE_STATIC_LIBS)  
    endif(NOT MSVC)
else(BUILD_SHARED_LIBS)
    target_link_libraries(
      xir
      PUBLIC protobuf::libprotobuf Threads::Threads ${CMAKE_DL_LIBS}
      PUBLIC unilog::unilog)
endif(BUILD_SHARED_LIBS)
# Enable Export of public APIs
target_compile_definitions(${PROJECT_NAME} PRIVATE -DXIR_EXPORT)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DXIR_USE_DLL=1)
else(BUILD_SHARED_LIBS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DXIR_USE_DLL=0)
endif(BUILD_SHARED_LIBS)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
foreach(
  PUBLIC_HEADER
  xir/XirExport.hpp
  xir/op/op.hpp
  xir/op/op_def.hpp
  xir/attrs/attrs.hpp
  xir/attrs/attr_def.hpp
  xir/attrs/attr_expander.hpp
  xir/util/any.hpp
  xir/util/tool_function.hpp
  xir/util/data_type.hpp
  xir/graph/graph.hpp
  xir/graph/graph_template.hpp
  xir/graph/subgraph.hpp
  xir/tensor/tensor.hpp)
  get_filename_component(HEADER_PATH ${PUBLIC_HEADER} DIRECTORY)
  install(FILES ${PROJECT_SOURCE_DIR}/include/${PUBLIC_HEADER}
          DESTINATION include/${HEADER_PATH})
endforeach()
install(
  EXPORT ${PROJECT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/cmake/${PROJECT_NAME})
