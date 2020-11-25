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

project(usefultools VERSION 1.1.0 LANGUAGES C CXX)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include(XilinxCommon)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
  set(TOOL_NAME "xilinx_test_camera")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/test_camera.cpp)
  target_link_libraries(${TOOL_NAME} ${OpenCV_LIBS} vart::util glog::glog)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_read_version")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/read_version.c)
  target_link_libraries(${TOOL_NAME} dl)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_irps5401")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/irps5401.c)
  target_link_libraries(${TOOL_NAME} dl)
  #install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_upgrade-reboot")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/reboot.c)
  target_link_libraries(${TOOL_NAME} dl)
  install(TARGETS ${TOOL_NAME} DESTINATION sbin/xlnx)

  set(TOOL_NAME "xilinx_mem_read")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/mem_read.cpp)
  target_link_libraries(${TOOL_NAME} glog::glog)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_mem_write")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/mem_write.cpp)
  target_link_libraries(${TOOL_NAME} glog::glog)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_show_devices")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/test_xrt_device_handle.cpp)
  target_link_libraries(${TOOL_NAME} glog::glog vart::xrt-device-handle )
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_test_dpu_runner")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/test_dpu_runner.cpp)
  target_link_libraries(${TOOL_NAME} glog::glog vart::runner xir::xir)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_xmodel_to_svg")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/xmodel_to_svg.cpp)
  target_link_libraries(${TOOL_NAME} glog::glog xir::xir)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_xmodel_to_png")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/xmodel_to_png.cpp)
  target_link_libraries(${TOOL_NAME} glog::glog xir::xir)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)


  set(TOOL_NAME "xilinx_show_xmodel_kernel")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/show_xmodel_kernel.cpp)
  target_link_libraries(${TOOL_NAME} glog::glog xir::xir)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)


  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/settings.sh.in ${CMAKE_CURRENT_BINARY_DIR}/settings.sh @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/settings.sh DESTINATION .)

  add_executable(xilinx_mpic src/mpic.cpp)
  target_link_libraries(xilinx_mpic ${OpenCV_LIBS} glog::glog)
  install(TARGETS xilinx_mpic DESTINATION bin/xlnx)


  # if(HAVE_XRT)
  #   add_definitions(-I=/usr/include/xrt)
  #   add_executable(xrt_mem_read ${CMAKE_CURRENT_SOURCE_DIR}/src/xrt_mem_read.cpp)
  #   add_executable(xrt_mem_write ${CMAKE_CURRENT_SOURCE_DIR}/src/xrt_mem_write.cpp)
  #   target_link_libraries(xrt_mem_read glog::glog xrt_core)
  #   target_link_libraries(xrt_mem_write glog::glog xrt_core)
  #   install(TARGETS xrt_mem_read xrt_mem_write DESTINATION bin)

  #   add_executable(xrt_mem_pic ${CMAKE_CURRENT_SOURCE_DIR}/src/xrt_mem_pic.cpp)
  #   target_link_libraries(xrt_mem_pic glog::glog xrt_core  ${OpenCV_LIBS})
  #   install(TARGETS xrt_mem_pic DESTINATION bin)
  # endif(HAVE_XRT)

  #add_executable(xir2elf ${CMAKE_CURRENT_SOURCE_DIR}/src/xir2elf.cpp)
  #target_link_libraries(xir2elf xir::xir vart::util vart::elf-util glog::glog)
  #install(TARGETS xir2elf DESTINATION bin)

  add_executable(tf_wb_conv ${CMAKE_CURRENT_SOURCE_DIR}/src/tf_wb_conv.cpp)
  target_link_libraries(tf_wb_conv glog::glog)
  install(TARGETS tf_wb_conv DESTINATION bin)

endif()
