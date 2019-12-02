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
#
cmake_minimum_required(VERSION 3.5)
project(dpclassification VERSION 1.0.0 LANGUAGES C CXX)
set(OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui)
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  ## TODO: fix hard coded directory
  link_directories(/home/administrator/workspace/qtbase/lib/)
  list(APPEND OpenCV_LIBS Qt5Widgets Qt5Test Qt5Concurrent Qt5Gui Qt5Core)
else()
  list(APPEND OpenCV_LIBS pthread)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SYSROOT}/usr/share/cmake/")
include_directories("${CMAKE_SOURCE_DIR}/include")
include(XilinxCommon)
include(XilinxVersion)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/word_list.inc
  COMMAND bash -ex ${CMAKE_SOURCE_DIR}/src/convert_word_list.sh ${CMAKE_SOURCE_DIR}/src/word_list.txt ${CMAKE_BINARY_DIR}/word_list.inc
  DEPENDS ${CMAKE_SOURCE_DIR}/src/convert_word_list.sh ${CMAKE_SOURCE_DIR}/src/word_list.txt
  )
include_directories("${CMAKE_BINARY_DIR}")

add_library(${PROJECT_NAME}
  include/xilinx/ai/classification.hpp
  src/classification.cpp
  src/classification_imp.cpp
  src/classification_imp.hpp
  ${CMAKE_BINARY_DIR}/word_list.inc
  ${CMAKE_BINARY_DIR}/version.c
  )
set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}")
target_link_libraries(${PROJECT_NAME} xnnpp dpbase dpproto glog ${OpenCV_LIBS})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES include/xilinx/ai/classification.hpp DESTINATION include/xilinx/ai)

add_executable(test_classification test/test_classification.cpp)
add_executable(test_accuracy_classification test/test_accuracy_classification.cpp)
target_link_libraries(test_classification ${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(test_accuracy_classification ${PROJECT_NAME} ${OpenCV_LIBS} protobuf)


