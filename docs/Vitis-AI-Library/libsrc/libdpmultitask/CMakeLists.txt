#
# (c) Copyright 2018 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.
#
cmake_minimum_required(VERSION 3.5)
project(dpmultitask VERSION 1.0.0 LANGUAGES C CXX)
include_directories("${CMAKE_SOURCE_DIR}/include")
set(OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui protobuf)
set(Pthread_LIB )
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  ## TODO: fix hard coded directory
  link_directories(/home/administrator/workspace/qtbase/lib/)
  list(APPEND OpenCV_LIBS Qt5Widgets Qt5Test Qt5Concurrent Qt5Gui Qt5Core)
else()
  list(APPEND Pthread_LIB pthread)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SYSROOT}/usr/share/cmake/")
include(XilinxCommon)
include(XilinxVersion)

add_library(${PROJECT_NAME}
  include/xilinx/ai/multitask.hpp
  src/multitask.cpp
  src/multitask_imp.cpp
  src/multitask_imp.hpp
  ${CMAKE_BINARY_DIR}/version.c
  )
set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}")
target_link_libraries(${PROJECT_NAME} xnnpp dpproto dpbase dpmath glog ${Pthread_LIB} ${OpenCV_LIBS})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES include/xilinx/ai/multitask.hpp DESTINATION include/xilinx/ai)

add_executable(test_multitask test/test_multitask.cpp)
target_link_libraries(test_multitask ${PROJECT_NAME} ${OpenCV_LIBS})
add_executable(test_accuracy_multitask test/test_accuracy_multitask.cpp)
target_link_libraries(test_accuracy_multitask ${PROJECT_NAME} ${OpenCV_LIBS} protobuf)
