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
get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/VitisVersion.cmake)

add_library(${COMPONENT_NAME} SHARED
  src/dpu_runner.cpp
  src/error_code.cpp
  src/tensor_buffer.cpp
  src/vart_c_api.cpp
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
  include/vart/vart.h
  include/vart/tensor_buffer_allocator.hpp
  src/tensor_buffer_allocator.cpp
  include/vitis/ai/dpu_runner.hpp
  include/vitis/ai/runner.hpp
  include/vitis/ai/tensor.hpp
  include/vitis/ai/tensor_buffer.hpp
  )
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
set_target_properties(${COMPONENT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
  LIBRARY_OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME})
target_include_directories(${COMPONENT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# Add JSON Util library
find_library(LIB_JSON_C json-c )
if(NOT LIB_JSON_C)
	message(FATAL_ERROR "json-c not found")
endif()

# Add JSON library, needed by DPUV1
target_link_libraries(${COMPONENT_NAME} PUBLIC unilog::unilog xir::xir
  PRIVATE ${PROJECT_NAME}::util ${LIB_JSON_C} dl)

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT base
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib)
foreach(PUBLIC_HEADER
    #v1.3
    vart/runner_ext.hpp
    # v1.2 interface
    vart/vart.h
    vart/runner.hpp
    vart/tensor_buffer.hpp
    # v1.1 interface
    vitis/ai/tensor_buffer.hpp
    vitis/ai/dpu_runner.hpp
    vitis/ai/runner.hpp
    vitis/ai/tensor.hpp
    )
  get_filename_component(HEADER_PATH ${PUBLIC_HEADER} DIRECTORY)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/${PUBLIC_HEADER}
    COMPONENT base
    DESTINATION include/${HEADER_PATH})
endforeach()

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT base
  DESTINATION share/cmake/${PROJECT_NAME})

if(BUILD_TEST)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_executable(test_runner test/test_runner.cpp)
  target_link_libraries(test_runner ${COMPONENT_NAME} ${PROJECT_NAME}::util pthread)
  add_executable(test_allocator test/test_allocator.cpp)
  target_link_libraries(test_allocator ${COMPONENT_NAME} ${PROJECT_NAME}::util pthread)
  add_executable(test_tensor_buffer test/test_tensor_buffer.cpp)
  target_link_libraries(test_tensor_buffer ${COMPONENT_NAME} ${PROJECT_NAME}::util pthread)

endif()

if(BUILD_PYTHON)
  # TODO, rename runner to dpu
  set(MODULE_NAME vart)
  if(CMAKE_CROSSCOMPILING)
    find_package(Python3 REQUIRED COMPONENTS Development)
    ## pybind11 does not support cross compiling very well
    find_path(_PYBIND11_PATH pybind11 HINTS /usr/include/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}m)
    message("_PYBIND11_PATH is ${_PYBIND11_PATH}")
    if(NOT _PYBIND11_PATH)
      message(WARNING "PYBIND11 NOT FOUND. python extenions for vitis dpu runner will not be built.")
    else()
      get_filename_component(PYBIND11_PATH ${_PYBIND11_PATH} DIRECTORY)
      add_library(vitis-ai-runner-py SHARED python/runner_py_module.cpp)
      set_target_properties(vitis-ai-runner-py PROPERTIES
        INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include;${_PYBIND11_PATH};${CMAKE_SYSROOT}/usr/include/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}m"
        COMPILE_DEFINITIONS "MODULE_NAME=${MODULE_NAME}"
        PREFIX ""
        LIBRARY_OUTPUT_NAME "${MODULE_NAME}")
      target_link_libraries(vitis-ai-runner-py -lpython${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}m ${COMPONENT_NAME})
      install(TARGETS vitis-ai-runner-py
	    COMPONENT base
        LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
      install(FILES python/runner.py
        COMPONENT base
        DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
    endif(NOT _PYBIND11_PATH)
  else(CMAKE_CROSSCOMPILING)
    find_package(pybind11 REQUIRED)

    set(PYBIND11_CPP_STANDARD -std=c++14)
    set(PYBIND11_INSTALL OFF)
    set(PYBIND11_TEST OFF)
    set(PYBIND11_PYTHON_VERSION  3.)
    pybind11_add_module(vitis-ai-runner-py python/runner_py_module.cpp)
    set_target_properties(vitis-ai-runner-py PROPERTIES
      LIBRARY_OUTPUT_NAME "${MODULE_NAME}")
    set_property(TARGET vitis-ai-runner-py
      APPEND PROPERTY COMPILE_DEFINITIONS  "MODULE_NAME=${MODULE_NAME}")
    set_property(TARGET vitis-ai-runner-py
      APPEND PROPERTY LINK_LIBRARIES  "${COMPONENT_NAME}")
    install(TARGETS vitis-ai-runner-py
      COMPONENT base
      LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
    install(FILES python/runner.py
      COMPONENT base
      DESTINATION ${PYTHON_SITE_PACKAGES})
  endif(CMAKE_CROSSCOMPILING)


endif(BUILD_PYTHON)
