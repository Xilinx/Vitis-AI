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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DEPS_PUBLIC xir::xir unilog::unilog runner uuid util json-c)
set(DEPS_PRIVATE XRT::xrt_core xrt-device-handle buffer-object)

include_directories(/opt/xilinx/xrt/include)
include_directories(${PROJECT_SOURCE_DIR}/buffer-object/include)
include_directories(${PROJECT_SOURCE_DIR}/xrt-device-handle/src)
include_directories(${PROJECT_SOURCE_DIR}/dpu-controller/src)

add_definitions(-DENABLE_CLOUD)
aux_source_directory(src MY_PROJECT_SOURCES)
list(APPEND MY_PROJECT_SOURCES
     ${PROJECT_SOURCE_DIR}/dpu-controller/src/xrt_cu.hpp
     ${PROJECT_SOURCE_DIR}/dpu-controller/src/xrt_cu.cpp)

get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
add_library(${COMPONENT_NAME} ${MY_PROJECT_SOURCES})
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})

target_include_directories(
  ${COMPONENT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME}
             INTERFACE_LINK_DIRECTORIES "/opt/xilinx/xrt/lib"
             BUILD_RPATH
             "\$ORIGIN/../buffer-object;\$ORIGIN/../xrt-device-handle")

target_link_libraries(
  ${COMPONENT_NAME}
  PUBLIC ${DEPS_PUBLIC}
  PRIVATE ${DEPS_PRIVATE})

install(
  TARGETS ${COMPONENT_NAME}
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

if(BUILD_TEST)
  include_directories(${PROJECT_SOURCE_DIR}/xrnn-runner/src/)
  include_directories(${PROJECT_SOURCE_DIR}/runner/include)
  include_directories(${PROJECT_SOURCE_DIR}/runner/src)
  add_executable(test_xrnn_runner test/test_xrnn_runner.cpp)
  target_link_libraries(test_xrnn_runner runner xir::xir glog::glog util)

  add_executable(test_xrnn_runner_u50 test/test_xrnn_runner_u50.cpp)
  target_link_libraries(test_xrnn_runner_u50 runner xir::xir glog::glog util)

  add_executable(test_xrnn_runner_mt test/test_xrnn_runner_mt.cpp)
  target_link_libraries(test_xrnn_runner_mt runner xir::xir glog::glog util)
endif()
