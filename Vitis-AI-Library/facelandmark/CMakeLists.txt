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
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  ## TODO: fix hard coded directory
  link_directories(/home/administrator/workspace/qtbase/lib/)
  list(APPEND OpenCV_LIBS Qt5Widgets Qt5Test Qt5Concurrent Qt5Gui Qt5Core)
else()
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SYSROOT}/usr/share/cmake/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

include(XilinxCommon)
include(XilinxVersion)
#include(XilinxDpu)
add_library(${COMPONENT_NAME} SHARED
  include/vitis/ai/facelandmark.hpp
  src/facelandmark.cpp
  src/facelandmark_imp.hpp
  src/facelandmark_imp.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/version.c
  )
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
target_link_libraries(${COMPONENT_NAME}
  PUBLIC ${OpenCV_LIBS}
  PRIVATE xnnpp dpu_task model_config glog)

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

install(FILES include/vitis/ai/facelandmark.hpp
  DESTINATION include/vitis/ai)

add_executable(test_face_landmark test/test_face_landmark.cpp)
target_link_libraries(test_face_landmark ${COMPONENT_NAME}  ${OpenCV_LIBS})

add_executable(test_face_landmark_batch test/test_face_landmark_batch.cpp)
target_link_libraries(test_face_landmark_batch ${COMPONENT_NAME}  ${OpenCV_LIBS})

add_executable(test_face_landmark_accuracy test/test_accuracy_face_landmark.cpp)
target_link_libraries(test_face_landmark_accuracy ${COMPONENT_NAME}  ${OpenCV_LIBS} model_config)
