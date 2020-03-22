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
set(OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui)

project(usefultools VERSION 1.1.0 LANGUAGES C CXX)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include(XilinxCommon)

include_directories("${CMAKE_INSTALL_PREFIX}/opt/xilinx/")
link_directories("${CMAKE_INSTALL_PREFIX}/opt/xilinx/xrt/lib")
include_directories("/opt/xilinx/xrt/include")
link_directories("/opt/xilinx/xrt/lib")
include(CheckIncludeFileCXX)
get_property(CMAKE_REQUIRED_INCLUDES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# see https://stackoverflow.com/questions/47213356/cmake-using-corrext-c-standard-when-checking-for-header-files
set(CMAKE_REQUIRED_FLAGS "-std=c++14")
check_include_file_cxx(xclhal2.h HAVE_XRT_CLOUD) #xrt_cloud
check_include_file_cxx(xrt/xclhal2.h HAVE_XRT_EDGE) #xrt_edge
if(HAVE_XRT_CLOUD)
  set(HAVE_XRT on)
endif(HAVE_XRT_CLOUD)
if(HAVE_XRT_EDGE)
  set(HAVE_XRT on)
endif(HAVE_XRT_EDGE)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_C_STANDARD 99)
if (NOT DP_ROOTFS STREQUAL "")
  include_directories(${DP_ROOTFS}/include)
  link_directories(${DP_ROOTFS}/lib)
endif()
set(CMAKE_C_FLAGS "-Wall -Werror")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
  set(TOOL_NAME "xilinx_opendl")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/opendl.c)
  target_link_libraries(${TOOL_NAME} dl)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
  set(TOOL_NAME "xilinx_irps5401")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/irps5401.c)
  target_link_libraries(${TOOL_NAME} dl)
  #install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
  set(TOOL_NAME "xilinx_upgrade-reboot")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/reboot.c)
  target_link_libraries(${TOOL_NAME} dl)
  install(TARGETS ${TOOL_NAME} DESTINATION sbin/xlnx)
endif()

# set(TOOL_NAME "reg-config")
# add_executable(${TOOL_NAME} ${CMAKE_SOURCE_DIR}/src/dpregconfig.cpp ${CMAKE_SOURCE_DIR}/src/dpregconfig_imp.cpp ${CMAKE_SOURCE_DIR}/src/main.cpp)
# install(TARGETS ${TOOL_NAME} DESTINATION bin)
# install(FILES ${CMAKE_SOURCE_DIR}/zu_reg_config DESTINATION share)

set(TOOL_NAME "xilinx_mem_read")
add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/mem_read.cpp)
target_link_libraries(${TOOL_NAME} glog)
install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

set(TOOL_NAME "xilinx_mem_write")
add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/mem_write.cpp)
target_link_libraries(${TOOL_NAME} glog)
install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/settings.sh.in ${CMAKE_CURRENT_BINARY_DIR}/settings.sh @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/settings.sh DESTINATION .)

  add_executable(xilinx_mpic src/mpic.cpp)
  target_link_libraries(xilinx_mpic ${OpenCV_LIBS} glog)
  install(TARGETS xilinx_mpic DESTINATION bin/xlnx)

endif()

#set(TOOL_NAME "xilinx_verreg")
#add_executable(${TOOL_NAME} ${CMAKE_SOURCE_DIR}/src/verreg.cpp)
#target_link_libraries(${TOOL_NAME} glog)
#install(TARGETS ${TOOL_NAME} DESTINATION bin)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
  install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/src/vaitrace/vaitrace DESTINATION bin/xlnx)
  install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/src/vaitrace/hwinfo.py DESTINATION bin/xlnx)
  install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/src/vaitrace/xapm.py DESTINATION bin/xlnx)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/vaitrace/vaitrace_defcfg.json DESTINATION bin/xlnx)

  #install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/src/vaitrace/libxapm.so DESTINATION bin/xlnx)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/vaitrace/xapm/include")
  set(CMAKE_C_FLAGS "-fPIC")

  add_library(xapm SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vaitrace/xapm/apm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vaitrace/xapm/apm_shell.cpp
  )
  target_link_libraries(xapm pthread)
  install(TARGETS xapm DESTINATION bin/xlnx)
endif()

if(HAVE_XRT)
  add_definitions(-I=/usr/include/xrt)
  add_executable(xrt_mem_read ${CMAKE_CURRENT_SOURCE_DIR}/src/xrt_mem_read.cpp)
  add_executable(xrt_mem_write ${CMAKE_CURRENT_SOURCE_DIR}/src/xrt_mem_write.cpp)
  target_link_libraries(xrt_mem_read glog xrt_core)
  target_link_libraries(xrt_mem_write glog xrt_core)
  install(TARGETS xrt_mem_read xrt_mem_write DESTINATION bin)

  add_executable(xrt_mem_pic ${CMAKE_CURRENT_SOURCE_DIR}/src/xrt_mem_pic.cpp)
  target_link_libraries(xrt_mem_pic glog xrt_core  ${OpenCV_LIBS})
  install(TARGETS xrt_mem_pic DESTINATION bin)
endif(HAVE_XRT)


add_executable(tf_wb_conv ${CMAKE_CURRENT_SOURCE_DIR}/src/tf_wb_conv.cpp)
target_link_libraries(tf_wb_conv glog)
install(TARGETS tf_wb_conv DESTINATION bin)

find_package(vart COMPONENTS elf-util REQUIRED)
find_package(xir REQUIRED)
add_executable(xir2elf ${CMAKE_CURRENT_SOURCE_DIR}/src/xir2elf.cpp)
target_link_libraries(xir2elf xir vart::elf-util glog)
install(TARGETS xir2elf DESTINATION bin)
