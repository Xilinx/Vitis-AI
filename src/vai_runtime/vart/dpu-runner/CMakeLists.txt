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

link_directories(${CMAKE_INSTALL_PREFIX}/lib)
set(DEPS_PUBLIC runner)
set(DEPS_PRIVATE xir::xir unilog::unilog dpu-controller xrt-device-handle
                 runner-assistant trace)

if(XRT_FOUND)
  list(APPEND DEPS_PRIVATE XRT::xrt_core XRT::xrt_coreutil)
endif(XRT_FOUND)

set(MY_PROJECT_SOURCES
    src/dpu_kernel.cpp
    src/dpu_kernel.hpp
    src/dpu_reg.hpp
    src/dpu_runner_base_imp.cpp
    src/dpu_runner_base_imp.hpp
    src/dpu_session_base_imp.cpp
    src/dpu_session_base_imp.hpp
    src/dpu_session.cpp
    src/dpu_session.hpp
    src/graph_holder.cpp
    src/graph_holder.hpp
    src/my_tensor.hpp
    src/vitis_dpu_runner_factory.cpp
    src/imp/dpu_core.hpp
    src/imp/dpu_runner_hbm.cpp
    src/imp/dpu_runner_hbm.hpp
    src/imp/dpu_kernel_ddr.cpp
    src/imp/dpu_cloud.hpp
    src/imp/dpu_kernel_ddr.hpp
    src/imp/dpu_kernel_hbm.cpp
    src/imp/dpu_session_imp.hpp
    src/imp/dpu_runner_ddr.hpp
    src/imp/hbm_config.cpp
    src/imp/hbm_config.hpp
    src/imp/hbm_manager.cpp
    src/imp/hbm_manager.hpp
    src/imp/hbm_manager_imp.cpp
    src/imp/hbm_manager_imp.hpp
    src/imp/hbm_manager_vec_imp.cpp
    src/imp/hbm_manager_vec_imp.hpp
    src/imp/dpu_kernel_hbm.hpp
    src/imp/dpu_runner_ddr.cpp
    src/imp/dpu_session_imp.cpp
    src/imp/dpu_core.cpp)

list(APPEND MY_PROJECT_SOURCES ${IMP_SRCS})
list(APPEND DEPS_PUBLIC buffer-object)

if(NOT IS_EDGE)
  list(APPEND DEPS_PRIVATE uuid mem-manager)
endif()

add_library(${COMPONENT_NAME} ${MY_PROJECT_SOURCES}
                              ${CMAKE_CURRENT_BINARY_DIR}/version.c)
if(IS_EDGE)
  target_compile_definitions(${COMPONENT_NAME} PRIVATE "-DIS_EDGE=1")
else(IS_EDGE)
  target_compile_definitions(${COMPONENT_NAME} PRIVATE "-DIS_EDGE=0")
endif(IS_EDGE)
target_include_directories(
  ${COMPONENT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME})
target_link_libraries(
  ${COMPONENT_NAME}
  PUBLIC ${DEPS_PUBLIC}
  PRIVATE ${DEPS_PRIVATE})

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT dpu
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(
  FILES include/vart/dpu/vitis_dpu_runner_factory.hpp
  COMPONENT dpu
  DESTINATION include/vart/dpu)

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT dpu
  DESTINATION share/cmake/${PROJECT_NAME})

# install(DIRECTORY samples DESTINATION share/vart)

if(BUILD_TEST)
  add_subdirectory(test)
  if(NOT IS_EDGE)
    add_subdirectory(test_cloud)
  endif(NOT IS_EDGE)
endif()
