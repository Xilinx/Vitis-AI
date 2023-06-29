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
get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
add_library(${COMPONENT_NAME} src/softmax_runner.cpp
                              include/vart/softmax_runner.hpp)
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
target_link_libraries(
  ${COMPONENT_NAME} ${PROJECT_NAME}::runner ${PROJECT_NAME}::mem-manager
  dpu-controller runner-assistant)
target_include_directories(
  ${COMPONENT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)
set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME})
install(
  TARGETS ${COMPONENT_NAME}
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
foreach(PUBLIC_HEADER vart/softmax_runner.hpp)
  get_filename_component(HEADER_PATH ${PUBLIC_HEADER} DIRECTORY)
  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/${PUBLIC_HEADER}
    COMPONENT base
    DESTINATION include/${HEADER_PATH})
endforeach()

add_executable(test_softmax_runner test/test_softmax_runner.cpp)
target_link_libraries(test_softmax_runner runner ${COMPONENT_NAME}
                      ${PROJECT_NAME}::util)
