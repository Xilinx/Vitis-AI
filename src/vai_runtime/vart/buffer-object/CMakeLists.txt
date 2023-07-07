#
# Copyright 2019 Xilinx Inc.
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
get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/VitisVersion.cmake)

set(MY_PROJECT_SOURCES
    include/xir/buffer_object.hpp include/xir/buffer_object_export.hpp
    src/buffer_object.cpp include/xir/device_memory.hpp src/device_memory.cpp)
set(MY_PROJECT_DEPS glog::glog)

if(MSVC)

else(MSVC)
  # dpcma is linux only.
  list(
    APPEND
    MY_PROJECT_SOURCES
    src/buffer_object_map.cpp
    src/buffer_object_map.hpp
    src/buffer_object_fd.cpp
    src/buffer_object_fd.hpp
    src/buffer_object_dpcma.hpp
    src/buffer_object_dpcma.cpp)
endif(MSVC)
if(XRT_CLOUD_FOUND)
  add_definitions(-DENABLE_CLOUD)
endif(XRT_CLOUD_FOUND)
list(APPEND MY_PROJECT_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/src/buffer_object_view.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/buffer_object_view.hpp)

if(XRT_FOUND)
  list(APPEND MY_PROJECT_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/src/buffer_object_xrt_imp.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/src/buffer_object_xrt_imp.cpp)
  list(APPEND MY_PROJECT_DEPS xrt-device-handle)
  list(APPEND MY_PROJECT_DEPS XRT::xrt_core XRT::xrt_coreutil)
endif(XRT_FOUND)

# optional implementation on QNX platform
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  list(APPEND MY_PROJECT_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/src/buffer_object_qnx.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/src/buffer_object_qnx.cpp)
  list(APPEND MY_PROJECT_DEPS cache)
endif()

if(XRT_CLOUD_FOUND)
  list(APPEND MY_PROJECT_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/src/device_memory_cloud.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/src/device_memory_cloud.cpp)
endif(XRT_CLOUD_FOUND)

if(IS_EDGE)
  list(APPEND MY_PROJECT_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/src/device_memory_edge.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/src/device_memory_edge.cpp)
endif(IS_EDGE)

add_library(${COMPONENT_NAME} ${MY_PROJECT_SOURCES}
                              ${CMAKE_CURRENT_BINARY_DIR}/version.c)
target_compile_definitions(${COMPONENT_NAME} PRIVATE VART_BUFFER_OBJECT_EXPORT)
if(IS_EDGE)
  target_compile_definitions(${COMPONENT_NAME} PRIVATE "-DIS_EDGE=1")
else(IS_EDGE)
  target_compile_definitions(${COMPONENT_NAME} PRIVATE "-DIS_EDGE=0")
endif(IS_EDGE)

add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})

set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME}
             BUILD_RPATH "\$ORIGIN/../xrt-device-handle")
target_link_libraries(
  ${COMPONENT_NAME}
  PRIVATE ${MY_PROJECT_DEPS}
  PUBLIC util)
target_include_directories(
  ${COMPONENT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

install(
  TARGETS ${COMPONENT_NAME}
  COMPONENT dpu
  EXPORT ${COMPONENT_NAME}-targets
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib)
install(
  FILES include/xir/buffer_object.hpp
        include/xir/device_memory.hpp
        # include/xir/buffer_object_manager.hpp
        # include/xir/buffer_object_manager_store.hpp
        # include/xir/device_scheduler.hpp
  COMPONENT dpu
  DESTINATION include/xir)

install(
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT dpu
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/cmake/${PROJECT_NAME})

if(BUILD_TEST)
  link_directories(${CMAKE_CURRENT_BINARY_DIR}/../xrt-device-handle/)
  add_executable(test_buffer_object test/test_buffer_object.cpp)
  target_link_libraries(test_buffer_object ${COMPONENT_NAME})
  # add_executable(test_device_scheduler test/test_device_scheduler.cpp)
  # target_link_libraries(test_device_scheduler ${PROJECT_NAME})
endif()

add_executable(mem_write test/mem_write.cpp)
target_link_libraries(mem_write ${COMPONENT_NAME} glog::glog)
add_executable(mem_read test/mem_read.cpp)
target_link_libraries(mem_read ${COMPONENT_NAME} glog::glog)
add_executable(mem_save test/mem_save.cpp)
target_link_libraries(mem_save ${COMPONENT_NAME} glog::glog)
add_executable(dump_op_weights test/dump_op_weights.cpp)
target_link_libraries(dump_op_weights ${COMPONENT_NAME} glog::glog)
add_executable(dump_op_weights_2 test/dump_op_weights_2.cpp)
target_link_libraries(dump_op_weights_2 ${COMPONENT_NAME} glog::glog)

install(
  TARGETS mem_write mem_read mem_save dump_op_weights dump_op_weights_2
  COMPONENT dpu
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib)
