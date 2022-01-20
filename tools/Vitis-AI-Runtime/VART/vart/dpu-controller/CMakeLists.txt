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

get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/VitisVersion.cmake)

set(MY_PROJECT_SOURCES include/xir/dpu_controller.hpp src/dpu_controller.cpp)
set(SFMX_SOURCES include/xir/sfm_controller.hpp src/sfm_controller.cpp)
set(MY_PROJECT_DEPS glog::glog)

# the qnx_dpu implementation
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  list(APPEND MY_PROJECT_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_controller_qnx.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_controller_qnx.hpp)
  list(APPEND MY_PROJECT_DEPS xlnx_qnx_dpu)
elseif(IS_EDGE)
  list(APPEND MY_PROJECT_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_controller_dnndk.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_controller_dnndk.hpp)
endif()

option(HAVE_SOFTMAX "enable softmax or not" on)

if(XRT_FOUND)
  list(APPEND MY_PROJECT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/xrt_cu.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/src/xrt_cu.cpp)
endif(XRT_FOUND)
if(XRT_FOUND)
  list(
    APPEND
    MY_PROJECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_control_xrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_control_xrt_edge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_control_xrt_edge.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_control_xrt_xv_dpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_control_xrt_xv_dpu.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_control_xrt_cloud.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu_control_xrt_cloud.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/xir/sfm_controller.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sfm_control_xrt_edge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sfm_control_xrt_edge.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sfm_controller.cpp)
  list(APPEND MY_PROJECT_DEPS xrt-device-handle XRT::xrt_core)
else(XRT_FOUND)
  if(MSVC)
    message(WARNING "SFMC controller is not implemented on MSVC platform")
    list(APPEND MY_PROJECT_SOURCES
         ${CMAKE_CURRENT_SOURCE_DIR}/src/sfm_controller_no_imp.cpp)
  else(MSVC)
    if(IS_EDGE)
      list(APPEND MY_PROJECT_SOURCES
           ${CMAKE_CURRENT_SOURCE_DIR}/src/sfm_controller_dnndk.cpp)
    else(IS_EDGE)
      message(WARNING "SFMC controller is not supported on cloud platform")
      list(APPEND MY_PROJECT_SOURCES
           ${CMAKE_CURRENT_SOURCE_DIR}/src/sfm_controller_no_imp.cpp)
    endif(IS_EDGE)
  endif(MSVC)
endif(XRT_FOUND)
include_directories(${PROJECT_SOURCE_DIR}/buffer-object/include)

add_library(${COMPONENT_NAME} SHARED ${MY_PROJECT_SOURCES} ${SFMX_SOURCES}
                                     ${CMAKE_CURRENT_BINARY_DIR}/version.c)
if(XRT_FOUND)
  target_compile_definitions(${COMPONENT_NAME} PRIVATE XRT_FOUND)
endif(XRT_FOUND)

set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             LIBRARY_OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME}
             BUILD_RPATH
             "\$ORIGIN/../buffer-object;\$ORIGIN/../xrt-device-handle")
target_link_libraries(
  ${COMPONENT_NAME}
  PRIVATE ${MY_PROJECT_DEPS} buffer-object
  PUBLIC util
  PRIVATE trace)
target_include_directories(
  ${COMPONENT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT dpu
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(
  FILES include/xir/dpu_controller.hpp include/xir/sfm_controller.hpp
  COMPONENT dpu
  DESTINATION include/xir)

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT dpu
  DESTINATION share/cmake/${PROJECT_NAME})

add_executable(show_dpu test/show_dpu.cpp)
target_link_libraries(show_dpu ${COMPONENT_NAME} glog::glog)
install(TARGETS show_dpu DESTINATION bin)

if(XRT_CLOUD_FOUND)
  set(ENABLE_CLOUD TRUE)
  target_compile_definitions(${COMPONENT_NAME} PRIVATE -DENABLE_CLOUD)
endif(XRT_CLOUD_FOUND)

if(BUILD_TEST)
  add_executable(test_softmax test/test_softmax.cpp)
  target_link_libraries(test_softmax ${COMPONENT_NAME} glog::glog)
  add_executable(test_softmax_mt test/test_softmax_mt.cpp)
  target_link_libraries(test_softmax_mt ${COMPONENT_NAME} glog::glog)
  install(TARGETS test_softmax_mt test_softmax DESTINATION bin)

  if(XRT_FOUND)
    add_executable(test_dpu_controller test/test_dpu_controller.cpp)
    target_link_libraries(test_dpu_controller ${COMPONENT_NAME} XRT::xrt_core
                          xrt-device-handle glog::glog)

    add_executable(test_dpu_controller_cloud test/test_dpu_controller_cloud.cpp)
    target_link_libraries(test_dpu_controller_cloud ${COMPONENT_NAME}
                          glog::glog)
  endif(XRT_FOUND)
endif()

set(PACKAGE_COMPONENTS
    "${PACKAGE_COMPONENTS};runner-assistant"
    PARENT_SCOPE)
add_subdirectory(runner-assistant)
