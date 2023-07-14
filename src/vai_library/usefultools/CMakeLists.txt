#
# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
  set(TOOL_NAME "xilinx_test_camera")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/test_camera.cpp)
  target_link_libraries(${TOOL_NAME} ${OpenCV_LIBS} vart::util glog::glog)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_test_video_writer")
  add_executable(${TOOL_NAME}
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/test_video_writer.cpp)
  target_link_libraries(${TOOL_NAME} ${OpenCV_LIBS} vart::util glog::glog)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_irps5401")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/irps5401.c)
  target_link_libraries(${TOOL_NAME} dl)
  # install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_upgrade-reboot")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/reboot.c)
  target_link_libraries(${TOOL_NAME} dl)
  install(TARGETS ${TOOL_NAME} DESTINATION sbin/xlnx)

  # run protoc
  set(protobuf_MODULE_COMPATIBLE
      ON
      CACHE BOOL "")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xir_graph_proto_v2.pb.cc
           ${CMAKE_CURRENT_BINARY_DIR}/xir_graph_proto_v2.pb.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/xir_graph_proto_v2.proto
    COMMAND
      protobuf::protoc ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path
      ${CMAKE_CURRENT_SOURCE_DIR}/src/
      ${CMAKE_CURRENT_SOURCE_DIR}/src/xir_graph_proto_v2.proto
    COMMENT "Running C++ protocol buffer compiler on proto xir_graph_proto_v2"
    VERBATIM)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/settings.sh.in
                 ${CMAKE_CURRENT_BINARY_DIR}/settings.sh @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/settings.sh DESTINATION .)

  add_executable(xilinx_mpic src/mpic.cpp)
  target_link_libraries(xilinx_mpic ${OpenCV_LIBS} glog::glog)
  install(TARGETS xilinx_mpic DESTINATION bin/xlnx)

  add_executable(tf_wb_conv ${CMAKE_CURRENT_SOURCE_DIR}/src/tf_wb_conv.cpp)
  target_link_libraries(tf_wb_conv glog::glog)
  install(TARGETS tf_wb_conv DESTINATION bin)

  if(XRT_FOUND)
    add_definitions(-DENABLE_XRT)
    if(XRT_CLOUD_FOUND)
      add_definitions(-DENABLE_CLOUD)
    endif(XRT_CLOUD_FOUND)
  endif(XRT_FOUND)

  set(XDPUTIL_SRC
      src/tools_extra_ops.hpp
      src/read_register.cpp
      src/xmodel_to_txt.cpp
      src/xilinx_version.cpp
      src/test_dpu_runner_mt.cpp
      src/xdputil_query.cpp
      src/test_op_run.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/xir_graph_proto_v2.pb.h)
  find_package(target-factory)

  if(BUILD_PYTHON)
    set(MODULE_NAME tools_extra_ops)
    vai_add_pybind11_module(${MODULE_NAME} MODULE_NAME ${MODULE_NAME}
                            ${XDPUTIL_SRC} src/tools_extra_ops.cpp)
    target_include_directories(${MODULE_NAME}
                               PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(
      ${MODULE_NAME}
      PRIVATE ${PROJECT_NAME}::cxir
              glog::glog
              protobuf::libprotobuf
              xir::xir
              crypto
              dl
              vart::buffer-object
              vart::dpu-runner
              vart::util
              vart::mem-manager
              vart::xrt-device-handle
              target-factory::target-factory
              ${PROJECT_NAME}::graph_runner
              ${PROJECT_NAME}::cpu_task
              gcc_atomic)
    if(XRT_FOUND)
      target_link_libraries(${MODULE_NAME} PRIVATE XRT::XRT
                                                   vart::xrt-device-handle)
    endif(XRT_FOUND)
    install(FILES python/xdputil.py DESTINATION ${PYTHON_SITE_PACKAGES})
    install(DIRECTORY python/xdputil_component
            DESTINATION ${PYTHON_SITE_PACKAGES})
    install(PROGRAMS python/xdputil DESTINATION bin)
  endif(BUILD_PYTHON)
endif()
