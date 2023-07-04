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
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/VitisVersion.cmake)

add_library(
  ${COMPONENT_NAME}
  include/vart/mm/host_flat_tensor_buffer.hpp src/host_flat_tensor_buffer.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/version.c)

add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME})
target_include_directories(
  ${COMPONENT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)
target_link_libraries(${COMPONENT_NAME} PUBLIC unilog::unilog xir::xir util
                                               runner)

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT base
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
foreach(PUBLIC_HEADER vart/mm/host_flat_tensor_buffer.hpp)
  get_filename_component(HEADER_PATH ${PUBLIC_HEADER} DIRECTORY)
  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/${PUBLIC_HEADER}
    COMPONENT base
    DESTINATION include/${HEADER_PATH})
endforeach()
install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT base
  DESTINATION share/cmake/${PROJECT_NAME})

# if(BUILD_TEST) include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)
# aux_source_directory(test TEST_SRCS) add_executable(test_runner ${TEST_SRCS})
# target_link_libraries(test_runner ${COMPONENT_NAME} pthread) endif()
