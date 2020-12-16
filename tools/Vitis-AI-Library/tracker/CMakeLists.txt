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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SYSROOT}/usr/share/cmake/")
include(XilinxCommon)
include(XilinxVersion)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
set(OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui)
add_library(${COMPONENT_NAME}
  include/xilinx/ai/tracker.hpp
  src/ftd/ftd_filter_linear.cpp  src/ftd/ftd_structure.cpp  src/ftd/ftd_trajectory.cpp
  src/ftd/ftd_filter_linear.hpp  src/ftd/ftd_structure.hpp  src/ftd/ftd_trajectory.hpp
  src/common.hpp   src/ring_queue.hpp  src/state_map.cpp  src/state_map.hpp
  src/tracker.cpp src/tracker_imp.cpp src/tracker_imp.hpp
  ${CMAKE_BINARY_DIR}/version.c
  )
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})

target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS} glog)
set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}")
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES include/xilinx/ai/tracker.hpp
  DESTINATION include/xilinx/ai/)
add_executable(test_track test/test_track.cpp)
target_link_libraries(test_track ${PROJECT_NAME} pthread)
