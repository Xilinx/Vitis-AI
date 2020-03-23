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

set(OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SYSROOT}/usr/share/cmake/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include(XilinxCommon)
include(XilinxVersion)
#include(XilinxDpu)

add_library(${COMPONENT_NAME} SHARED
  include/vitis/ai/openpose.hpp
  src/openpose.cpp
  src/openpose_imp.cpp
  src/openpose_imp.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/version.c
  )
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
target_link_libraries(${COMPONENT_NAME}
  PRIVATE xnnpp dpu_task
  PUBLIC ${OpenCV_LIBS} glog)
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

install(FILES include/vitis/ai/openpose.hpp DESTINATION include/vitis/ai)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
add_executable(test_open_pose test/test_open_pose.cpp)
add_executable(test_open_pose_batch test/test_open_pose_batch.cpp)
target_link_libraries(test_open_pose ${COMPONENT_NAME} ${OpenCV_LIBS})
target_link_libraries(test_open_pose_batch ${COMPONENT_NAME} ${OpenCV_LIBS})
add_executable(test_open_pose_accuracy test/test_accuracy_open_pose.cpp)
target_link_libraries(test_open_pose_accuracy ${COMPONENT_NAME} ${OpenCV_LIBS} -ljson-c model_config)
