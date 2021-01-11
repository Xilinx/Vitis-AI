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
add_library(${COMPONENT_NAME} SHARED
  include/vart/op_imp.h
  src/cpu_task.hpp
  src/cpu_task.cpp
  src/op_imp.cpp
  src/batch_tensor_buffer_view.hpp
  src/batch_tensor_buffer_view.cpp
  )
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
target_link_libraries(${COMPONENT_NAME} vart::runner ${PROJECT_NAME}::runner_helper vart::mem-manager dl)
target_include_directories(${COMPONENT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
set_target_properties(${COMPONENT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
  LIBRARY_OUTPUT_NAME  ${PROJECT_NAME}-${COMPONENT_NAME}
  )
install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  RUNTIME DESTINATION ${INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${INSTALL_LIB_DIR})

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${INSTALL_CMAKE_DIR})

macro(ADD_OP_IMP OP_TYPE)
  add_library(${OP_TYPE} SHARED ${ARGN})
  target_link_libraries(${OP_TYPE} ${COMPONENT_NAME})
  set_target_properties(${OP_TYPE} PROPERTIES
    LIBRARY_OUTPUT_NAME vart_op_imp_${OP_TYPE}
    )
  target_include_directories(${OP_TYPE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
  install(
    TARGETS ${OP_TYPE}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    )
endmacro()

add_op_imp (softmax ops/softmax/softmax.cpp)
add_op_imp (fix2float ops/fix2float/fix2float.cpp)
add_op_imp (fix ops/fix/fix.cpp)

add_executable (test_cpu_task test/test_cpu_task.cpp)
target_link_libraries (test_cpu_task vart::runner  ${PROJECT_NAME}::runner_helper  vart::util)
