
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
include(XilinxCommon)
include(XilinxVersion)
set (COMPONENT_NAME "general1")
add_library(${COMPONENT_NAME} SHARED
  include/vitis/ai/general.hpp
  src/general_adapter.hpp
  src/general.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/version.c
  )
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})

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

#configure_file(config.h.in ${CMAKE_BINARY_DIR}/config.h)
target_link_libraries(${COMPONENT_NAME} ${PROJECT_NAME}::model_config ${PROJECT_NAME}::refinedet ${PROJECT_NAME}::facefeature ${PROJECT_NAME}::segmentation ${PROJECT_NAME}::reid ${PROJECT_NAME}::multitask ${PROJECT_NAME}::medicalsegmentation ${PROJECT_NAME}::posedetect ${PROJECT_NAME}::platenum ${PROJECT_NAME}::platedetect ${PROJECT_NAME}::facelandmark ${PROJECT_NAME}::lanedetect ${PROJECT_NAME}::ssd ${PROJECT_NAME}::tfssd ${PROJECT_NAME}::yolov2 ${PROJECT_NAME}::yolov3  ${PROJECT_NAME}::classification ${PROJECT_NAME}::facedetect  ${PROJECT_NAME}::facedetectrecog glog::glog ${OpenCV_LIBS} protobuf::libprotobuf)

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  RUNTIME DESTINATION ${INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${INSTALL_LIB_DIR})
install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${INSTALL_CMAKE_DIR})
install(FILES include/vitis/ai/general.hpp
  DESTINATION include/vitis/ai)

add_executable(test_general test/test_general.cpp)
add_executable(test_general_batch test/test_general_batch.cpp)
add_executable(generate_proto test/generate_proto.cpp)
add_executable(compare_proto test/compare_proto.cpp)
add_executable(show_proto test/show_proto.cpp)
target_link_libraries(test_general ${COMPONENT_NAME} ${OpenCV_LIBS})
target_link_libraries(test_general_batch ${COMPONENT_NAME} ${OpenCV_LIBS})
target_link_libraries(generate_proto ${COMPONENT_NAME} ${OpenCV_LIBS})
target_link_libraries(compare_proto ${COMPONENT_NAME})
target_link_libraries(show_proto ${COMPONENT_NAME})
