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
add_library(${COMPONENT_NAME} src/vart_dummy_runner.cpp)
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
target_link_libraries(${COMPONENT_NAME} ${PROJECT_NAME}::runner
                      ${PROJECT_NAME}::mem-manager)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(${COMPONENT_NAME}
                             PUBLIC -DVART_DUMMY_RUNNER_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(${COMPONENT_NAME}
                             PUBLIC -DVART_DUMMY_RUNNER_USE_DLL=0)
endif(BUILD_SHARED_LIBS)

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

if(NOT MSVC)
  add_executable(test_dummy_runner test/test_dummy_runner.cpp)
  target_link_libraries(test_dummy_runner runner ${PROJECT_NAME}::util)
endif(NOT MSVC)

add_executable(test_dummy_runner_simple test/test_dummy_runner_simple.cpp)
if(BUILD_SHARED_LIBS)
  target_link_libraries(test_dummy_runner_simple runner ${PROJECT_NAME}::util)
else(BUILD_SHARED_LIBS)
  target_link_libraries(test_dummy_runner_simple runner ${PROJECT_NAME}::util
                        ${COMPONENT_NAME})
endif(BUILD_SHARED_LIBS)

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT dpu
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT dpu
  DESTINATION share/cmake/${PROJECT_NAME})
