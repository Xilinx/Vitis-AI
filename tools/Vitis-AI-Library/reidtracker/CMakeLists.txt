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

set(OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui)
set(SAMPLE_INATLL_PATH share/${PROJECT_NAME}/samples)

include(XilinxCommon)
include(XilinxVersion)

add_library(${COMPONENT_NAME} SHARED
  include/vitis/ai/reidtracker.hpp
  src/ftd/ftd_filter_linear.cpp  src/ftd/ftd_structure.cpp  src/ftd/ftd_trajectory.cpp
  src/ftd/ftd_filter_linear.hpp  src/ftd/ftd_structure.hpp  src/ftd/ftd_trajectory.hpp
  src/ftd/ftd_hungarian.hpp
  src/ftd/ftd_hungarian.cpp
  src/common.hpp   src/ring_queue.hpp  src/state_map.cpp  src/state_map.hpp
  src/tracker.cpp src/tracker_imp.cpp src/tracker_imp.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/version.c
  )
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
target_link_libraries(${COMPONENT_NAME}  ${OpenCV_LIBS} ${PROJECT_NAME}::reid pthread glog model_config)
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

install(FILES include/vitis/ai/reidtracker.hpp
  DESTINATION include/vitis/ai/)

add_executable(test_images_reidtracker test/test_images_reidtracker.cpp)
target_link_libraries(test_images_reidtracker ${COMPONENT_NAME} pthread ${PROJECT_NAME}::refinedet)
add_executable(test_accuracy_reidtracker test/test_accuracy_reidtracker.cpp)
target_link_libraries(test_accuracy_reidtracker ${COMPONENT_NAME} pthread ${PROJECT_NAME}::refinedet)
install(TARGETS test_accuracy_reidtracker DESTINATION ${SAMPLE_INATLL_PATH}/reidtracker/)
add_executable(test_video test/reidtracker.cpp)
target_link_libraries(test_video ${COMPONENT_NAME} ${OpenCV_LIBS} ${PROJECT_NAME}::refinedet ${PROJECT_NAME}::reid glog pthread)

find_path(DRM_PATH drm)
if(NOT ${DRM_PATH} EQUAL "DRM_PATH-NOTFOUND")
    set(CMAKE_REQUIRED_INCLUDES ${DRM_PATH}/drm)
endif()
check_include_file(xf86drm.h HAVE_DRM)
if(HAVE_DRM)
    message(STATUS "Using DRM.")
    include_directories(${DRM_PATH}/drm)
    add_definitions(-DUSE_DRM=1)
endif(HAVE_DRM)
if(HAVE_DRM)
  target_link_libraries(test_video drm)
endif(HAVE_DRM)


