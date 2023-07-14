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
option(USE_JSON_C "use libjson_c or not" OFF)
get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/VitisVersion.cmake)

add_library(
  ${COMPONENT_NAME}
  src/dpu_runner.cpp
  src/error_code.cpp
  src/tensor_buffer.cpp
  src/tensor_buffer_unowned_device.cpp
  src/runner_helper.hpp
  src/runner_helper.cpp
  src/runner_ext.cpp
  v1.1/dpu_runner.cpp
  v1.1/tensor_buffer.cpp
  v1.1/tensor.cpp
  v1.1/runner_adaptor.hpp
  v1.1/runner_adaptor.cpp
  v1.1/tensor_buffer_adaptor.hpp
  v1.1/tensor_buffer_adaptor.cpp
  v1.1/convert_tensor.hpp
  v1.1/convert_tensor.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/version.c
  include/vart/runner.hpp
  include/vart/runner_ext.hpp
  include/vart/tensor_buffer.hpp
  include/vart/tensor_buffer_unowned_device.hpp
  include/vitis/ai/dpu_runner.hpp
  include/vitis/ai/runner.hpp
  include/vitis/ai/tensor.hpp
  include/vitis/ai/tensor_buffer.hpp)
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})

if(BUILD_SHARED_LIBS)
  target_compile_definitions(${COMPONENT_NAME} PUBLIC -DVART_RUNNER_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(${COMPONENT_NAME} PUBLIC -DVART_RUNNER_USE_DLL=0)
endif(BUILD_SHARED_LIBS)
set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME})
target_include_directories(
  ${COMPONENT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)
# Add JSON Util library

if(USE_JSON_C)
  find_library(LIB_JSON_C json-c)
  if(NOT LIB_JSON_C)
    message(FATAL_ERROR "json-c not found")
  endif()
endif(USE_JSON_C)

if(USE_JSON_C)
  target_compile_definitions(${COMPONENT_NAME} PRIVATE "-DUSE_JSON_C=1")
else(USE_JSON_C)
  target_compile_definitions(${COMPONENT_NAME} PRIVATE "-DUSE_JSON_C=0")
endif(USE_JSON_C)

# Add JSON library, needed by DPUV1
target_link_libraries(
  ${COMPONENT_NAME}
  PUBLIC xir::xir
  PRIVATE ${PROJECT_NAME}::util unilog::unilog)

if(USE_JSON_C)
  target_link_libraries(${COMPONENT_NAME} PRIVATE ${LIB_JSON_C})
endif(USE_JSON_C)

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT base
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
foreach(
  PUBLIC_HEADER
  # v1.3
  vart/runner_ext.hpp
  # v1.2 interface
  vart/runner.hpp
  vart/tensor_buffer.hpp
  vart/tensor_buffer_unowned_device.hpp
  # v1.1 interface
  vitis/ai/tensor_buffer.hpp
  vitis/ai/dpu_runner.hpp
  vitis/ai/runner.hpp
  vitis/ai/tensor.hpp)
  get_filename_component(HEADER_PATH ${PUBLIC_HEADER} DIRECTORY)
  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/${PUBLIC_HEADER}
    COMPONENT base
    DESTINATION include/${HEADER_PATH})
endforeach()

# ONEHACK TM Expose runner_helper.hpp to the world as experimental
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/runner_helper.hpp
  COMPONENT base
  DESTINATION include/vart/experimental)

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT base
  DESTINATION share/cmake/${PROJECT_NAME})

if(BUILD_TEST)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)
  if(USE_JSON_C) # create_dpu_runner only available if USE_JSON_C
    add_executable(test_runner test/test_runner.cpp)
    target_link_libraries(test_runner ${COMPONENT_NAME} ${PROJECT_NAME}::util
                          pthread)
  endif(USE_JSON_C)
  add_executable(test_tensor_buffer test/test_tensor_buffer.cpp)
  target_link_libraries(test_tensor_buffer ${COMPONENT_NAME}
                        ${PROJECT_NAME}::util ${CMAKE_THREAD_LIBS_INIT})

endif()

if(BUILD_PYTHON)
  vai_add_pybind11_module(py_runner MODULE_NAME vart
                          python/runner_py_module.cpp)
  target_link_libraries(py_runner PRIVATE ${PROJECT_NAME}::util
                                          ${PROJECT_NAME}::${COMPONENT_NAME})
endif(BUILD_PYTHON)
