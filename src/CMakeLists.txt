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

# get the git repo branch and commit id and write them into config.hpp
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE PROJECT_GIT_BRANCH_NAME
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE PROJECT_GIT_COMMIT_ID
  OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(config.hpp.in config.hpp @ONLY)

# run protoc
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS target.proto)
set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_FLAGS -Wno-unused-variable)

# process target files
file(GLOB PROTOTXT_FILE_LIST ${CMAKE_SOURCE_DIR}/targets/*.prototxt)
set(PROCESS_ONE_FILE "sed -e '1i#begin' -e '$a#end' -e 's/\"/\\\\\\\"/g' -e 's/$/\\\\\\\\n/' | xargs echo -n")
foreach(PROTOTXT_FILE ${PROTOTXT_FILE_LIST})
  execute_process(
    COMMAND bash "-c" "cat ${PROTOTXT_FILE} | ${PROCESS_ONE_FILE}"
    OUTPUT_VARIABLE PROCESS_OUTPUT)
  set(TARGET_PROTOTXTS ${TARGET_PROTOTXTS}${PROCESS_OUTPUT})
endforeach()
configure_file(${CMAKE_SOURCE_DIR}/src/target_list.hpp.in target_list.hpp
  ESCAPE_QUOTES @ONLY)

# build library
aux_source_directory(. CPP_SRCS)
add_library(${PROJECT_NAME} SHARED ${PROTO_SRCS} ${CPP_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}")
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} PUBLIC protobuf::libprotobuf unilog::unilog glog::glog)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib)
install(
  FILES ${PROTO_HDRS} ${CMAKE_SOURCE_DIR}/include/vitis/ai/target_factory.hpp
  DESTINATION include/vitis/ai)
install(
  EXPORT ${PROJECT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/cmake/${PROJECT_NAME})
