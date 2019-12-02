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
project(dpfacedetect VERSION 1.0.0 LANGUAGES C CXX)
set(OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui protobuf)
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  ## TODO: fix hard coded directory
  link_directories(/home/administrator/workspace/qtbase/lib/)
  list(APPEND OpenCV_LIBS Qt5Widgets Qt5Test Qt5Concurrent Qt5Gui Qt5Core)
else()
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SYSROOT}/usr/share/cmake/")
include_directories("${CMAKE_SOURCE_DIR}/include")

include(XilinxCommon)
include(XilinxVersion)
add_library(${PROJECT_NAME}
  include/xilinx/ai/facedetect.hpp
  src/facedetect.cpp
  src/detect_imp.hpp
  src/detect_imp.cpp
  ${CMAKE_BINARY_DIR}/version.c
  )
target_link_libraries(${PROJECT_NAME}
  dpbase # dpu_task
	xnnpp
  dpproto
  ${OpenCV_LIBS} # cv::resize
  glog
  )
set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}")
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES include/xilinx/ai/facedetect.hpp
  DESTINATION include/xilinx/ai)

add_executable(test_densebox test/test_densebox.cpp)
target_link_libraries(test_densebox ${PROJECT_NAME})

add_executable(test_accuracy_densebox test/test_accuracy_densebox.cpp)
target_link_libraries(test_accuracy_densebox ${PROJECT_NAME} dpmath)


