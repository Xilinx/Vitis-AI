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
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include(XilinxCommon)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
  set(TOOL_NAME "xilinx_test_camera")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/test_camera.cpp)
  target_link_libraries(${TOOL_NAME} ${OpenCV_LIBS} vart::util glog::glog)
  install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_irps5401")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/irps5401.c)
  target_link_libraries(${TOOL_NAME} dl)
  #install(TARGETS ${TOOL_NAME} DESTINATION bin/xlnx)

  set(TOOL_NAME "xilinx_upgrade-reboot")
  add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/reboot.c)
  target_link_libraries(${TOOL_NAME} dl)
  install(TARGETS ${TOOL_NAME} DESTINATION sbin/xlnx)

  # run protoc
  set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "")
  add_custom_command(
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/xir_graph_proto_v2.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/xir_graph_proto_v2.pb.h
    DEPENDS   ${CMAKE_CURRENT_SOURCE_DIR}/src/xir_graph_proto_v2.proto
    COMMAND  protobuf::protoc
    ARGS --cpp_out  ${CMAKE_CURRENT_BINARY_DIR}
    --proto_path ${CMAKE_CURRENT_SOURCE_DIR}/src/
    ${CMAKE_CURRENT_SOURCE_DIR}/src/xir_graph_proto_v2.proto
    COMMENT "Running C++ protocol buffer compiler on proto xir_graph_proto_v2"
    VERBATIM )

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/settings.sh.in ${CMAKE_CURRENT_BINARY_DIR}/settings.sh @ONLY)
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
  
  SET(XDPUTIL_SRC 
    ${CMAKE_CURRENT_BINARY_DIR}/xir_graph_proto_v2.pb.h
    src/tools_extra_ops.hpp 
    src/read_register.cpp 
    src/xmodel_to_txt.cpp 
    src/xilinx_version.cpp
    src/test_dpu_runner_mt.cpp
  )
  find_package(target-factory)
  
  if(BUILD_PYTHON)
    # TODO, rename runner to dpu
    set(MODULE_NAME tools_extra_ops)
    if(CMAKE_CROSSCOMPILING)
      find_package(Python3 REQUIRED COMPONENTS Development)
      find_path(_PYBIND11_PATH pybind11 HINTS /usr/include/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}m)
      message("_PYBIND11_PATH is ${_PYBIND11_PATH}")
      if(NOT _PYBIND11_PATH)
        message(WARNING "PYBIND11 NOT FOUND. python extenions for vitis dpu runner will not be built.")
      else()
        get_filename_component(PYBIND11_PATH ${_PYBIND11_PATH} DIRECTORY)
        add_library(${MODULE_NAME} SHARED ${XDPUTIL_SRC} src/tools_extra_ops.cpp )
        set_target_properties(${MODULE_NAME} PROPERTIES
          INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_CURRENT_SOURCE_DIR}/include;${_PYBIND11_PATH};${CMAKE_SYSROOT}/usr/include/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}m"
          COMPILE_DEFINITIONS "MODULE_NAME=${MODULE_NAME}"
          PREFIX ""
          LIBRARY_OUTPUT_NAME "${MODULE_NAME}")
        target_link_libraries(${MODULE_NAME} PRIVATE -lpython${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}m ${COMPONENT_NAME})
        target_link_libraries(${MODULE_NAME} PRIVATE glog::glog protobuf::libprotobuf xir::xir crypto dl vart::buffer-object vart::dpu-runner vart::util vart::mem-manager target-factory::target-factory)
        if(XRT_FOUND)
          target_link_libraries (${MODULE_NAME} PRIVATE XRT::XRT vart::xrt-device-handle )
        endif(XRT_FOUND)
        install(TARGETS ${MODULE_NAME}
	        COMPONENT base
          LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
        install(FILES python/xdputil.py DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
        install(DIRECTORY python/xdputil_component DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
      endif(NOT _PYBIND11_PATH)
    else(CMAKE_CROSSCOMPILING)
      find_package(pybind11 REQUIRED)
      set(PYBIND11_CPP_STANDARD -std=c++14)
      set(PYBIND11_INSTALL OFF)
      set(PYBIND11_TEST OFF)
      set(PYBIND11_PYTHON_VERSION  3)
      pybind11_add_module(${MODULE_NAME} SHARED ${XDPUTIL_SRC} src/tools_extra_ops.cpp )
      target_link_libraries (${MODULE_NAME} PRIVATE ${COMPONENT} )
      target_link_libraries (${MODULE_NAME} PRIVATE ${PYTHON_LIBRARIES})
      target_link_libraries (${MODULE_NAME} PRIVATE glog::glog protobuf::libprotobuf xir::xir crypto dl vart::buffer-object vart::dpu-runner vart::util vart::mem-manager target-factory::target-factory)
      if(XRT_FOUND)
        target_link_libraries (${MODULE_NAME} PRIVATE XRT::XRT vart::xrt-device-handle )
      endif(XRT_FOUND)

      set_target_properties(${MODULE_NAME} PROPERTIES LIBRARY_OUTPUT_NAME "${MODULE_NAME}")

      set_property(TARGET ${MODULE_NAME} APPEND PROPERTY COMPILE_DEFINITIONS  "MODULE_NAME=${MODULE_NAME}")
      set_property(TARGET ${MODULE_NAME} APPEND PROPERTY INCLUDE_DIRECTORIES  "${PYTHON_INCLUDE_DIRS}")
      get_property(TMP TARGET ${MODULE_NAME} PROPERTY LINK_LIBRARIES)
      message("TMP is ${TMP}")
      install(TARGETS ${MODULE_NAME} COMPONENT base LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
      install(FILES python/xdputil.py DESTINATION ${PYTHON_SITE_PACKAGES})
      install(DIRECTORY python/xdputil_component DESTINATION ${PYTHON_SITE_PACKAGES})
    endif(CMAKE_CROSSCOMPILING)
    install(PROGRAMS python/xdputil DESTINATION bin)
  endif(BUILD_PYTHON)

endif()


