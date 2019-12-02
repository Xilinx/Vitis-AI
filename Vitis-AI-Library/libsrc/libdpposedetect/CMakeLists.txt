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
project(dpposedetect VERSION 1.0.0 LANGUAGES C CXX)
include_directories("${CMAKE_SOURCE_DIR}/include")
set(OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui)
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  ## TODO: fix hard coded directory
  link_directories(/home/administrator/workspace/qtbase/lib/)
  list(APPEND OpenCV_LIBS Qt5Widgets Qt5Test Qt5Concurrent Qt5Gui Qt5Core)
else()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SYSROOT}/usr/share/cmake/")
include(XilinxCommon)
include(XilinxVersion)

add_library(${PROJECT_NAME}
  include/xilinx/ai/posedetect.hpp
  src/posedetect.cpp
  src/posedetect_imp.cpp
  src/posedetect_imp.hpp
  ${CMAKE_BINARY_DIR}/version.c
  )

set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}")
target_link_libraries(${PROJECT_NAME} dpproto xnnpp dpbase dpmath ${OpenCV_LIBS} glog)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES include/xilinx/ai/posedetect.hpp DESTINATION include/xilinx/ai)

add_executable(test_pose_detect test/test_pose_detect.cpp)
target_link_libraries(test_pose_detect ${PROJECT_NAME} ${OpenCV_LIBS})
add_executable(test_accuracy_pose_detect test/test_accuracy_pose_detect.cpp)
target_link_libraries(test_accuracy_pose_detect ${PROJECT_NAME} ${OpenCV_LIBS})


