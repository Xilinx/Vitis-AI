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
find_package(vart REQUIRED COMPONENTS util xrt-device-handle dpu-controller)
set(OpenCV_LIBS glog opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui)
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  ## TODO: fix hard coded directory
  link_directories(/home/administrator/workspace/qtbase/lib/)
  list(APPEND OpenCV_LIBS Qt5Widgets Qt5Test Qt5Concurrent Qt5Gui Qt5Core)
else()
  list(APPEND OpenCV_LIBS pthread)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include(XilinxCommon)
include(XilinxVersion)
include(CheckIncludeFileCXX)
check_include_file_cxx(arm_neon.h HAVE_NEON)
if(HAVE_NEON)
  add_definitions(-DENABLE_NEON)
endif(HAVE_NEON)

add_definitions(-ggdb)
set(LIBRARY_SOURCE
  src/exp_neon.c
  src/softmax.cpp
  src/ssd_normalizer_c.cpp
  src/ssd_normalizer_neon.cpp
  src/max_index.cpp
  src/yuv2bgr.cpp
  src/tiling.cpp
#  include/vitis/fp16_base.hpp
#  include/vitis/softmax.hpp
  include/vitis/ai/ssd_normalizer.hpp
  include/vitis/ai/math.hpp
  include/vitis/ai/max_index.hpp
  include/vitis/ai/expand_and_align.hpp
  include/vitis/ai/globalavepool.hpp
  include/vitis/ai/image_util.hpp
  include/vitis/ai/preprocess.hpp
  src/expand_and_align.cpp
  src/globalavepool.cpp
  src/image_util.cpp
  src/preprocess_c.cpp
  src/preprocess_neon.cpp
)
if(HAVE_NEON)
  list(APPEND LIBRARY_SOURCE
    src/softmax_table.hpp
    src/softmax_table.cpp
  )
endif()

add_library(${COMPONENT_NAME} SHARED ${LIBRARY_SOURCE})
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
set_target_properties(${COMPONENT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
  LIBRARY_OUTPUT_NAME  ${PROJECT_NAME}-${COMPONENT_NAME}
  )
target_link_libraries(${COMPONENT_NAME} glog vart::util vart::dpu-controller)
install(FILES
  include/vitis/ai/math.hpp
#  include/vitis/fp16_base.hpp
#  include/vitis/softmax.hpp
  include/vitis/ai/ssd_normalizer.hpp
  include/vitis/ai/max_index.hpp
  include/vitis/ai/expand_and_align.hpp
  include/vitis/ai/globalavepool.hpp
  include/vitis/ai/image_util.hpp
  include/vitis/ai/preprocess.hpp
  DESTINATION include/vitis/ai)

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  RUNTIME DESTINATION ${INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${INSTALL_LIB_DIR})

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${INSTALL_CMAKE_DIR})


set(GTEST_DEPS gtest_main gtest)
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
else()
  list(APPEND GTEST_DEPS pthread)
endif()

foreach(TMP test_normalize)
  add_executable(${TMP} test/${TMP}.cpp)
  target_link_libraries(${TMP} ${COMPONENT_NAME} ${GTEST_DEPS} )
endforeach(TMP)


if(HAVE_NEON)
  add_executable(test_max_index test/test_max_index.cpp)
  target_link_libraries(test_max_index ${COMPONENT_NAME} ${GTEST_DEPS})

  add_executable(test_softmax_my test/test_softmax_my.cpp)
  target_link_libraries(test_softmax_my ${COMPONENT_NAME} ${OpenCV_LIBS} glog)

  add_executable(bgr_to_yuv420 test/bgr_to_yuv420.cpp)
  target_link_libraries(bgr_to_yuv420 ${OpenCV_LIBS} glog)

  add_executable(test_softmax_table test/test_softmax_table.cpp)
  target_link_libraries(test_softmax_table ${COMPONENT_NAME} ${OpenCV_LIBS} glog)


endif(HAVE_NEON)

add_executable(test_softmax_1 test/test_softmax_1.cpp)
target_link_libraries(test_softmax_1 ${COMPONENT_NAME} ${OpenCV_LIBS} glog)
