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
cmake_minimum_required(VERSION 3.5)
get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

check_include_file_cxx(eigen3/Eigen/Dense HAVE_EIGEN)

include(XilinxVersion)
add_library(${COMPONENT_NAME} SHARED
  include/vitis/ai/facedetectrecog.hpp
  src/facedetectrecog.cpp
  src/facedetectrecog_imp.hpp
  src/facedetectrecog_imp.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/version.c
  )
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})

#target_link_libraries(${COMPONENT_NAME} ${PROJECT_NAME}::xnnpp ${PROJECT_NAME}::dpu_task)
target_link_libraries(${COMPONENT_NAME} PUBLIC ${PROJECT_NAME}::facedetect ${PROJECT_NAME}::facelandmark ${PROJECT_NAME}::facefeature ${PROJECT_NAME}::math
                                        PRIVATE xir::xir)

target_include_directories(${COMPONENT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(${COMPONENT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
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
install(FILES include/vitis/ai/facedetectrecog.hpp
  DESTINATION include/vitis/ai)

add_executable(test_facedetectrecog test/test_facedetectrecog.cpp)
target_link_libraries(test_facedetectrecog ${COMPONENT_NAME})
add_executable(test_compare_l20 test/test_compare_l20.cpp)
target_link_libraries(test_compare_l20 ${COMPONENT_NAME})
#
add_executable(test_facedetectrecog_batch test/test_facedetectrecog_batch.cpp)
target_link_libraries(test_facedetectrecog_batch ${COMPONENT_NAME})

