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

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")

if(XRT_CLOUD_FOUND)
  add_definitions(-DENABLE_CLOUD)
endif(XRT_CLOUD_FOUND)

find_package(Boost COMPONENTS filesystem)
# find_package(butler PATHS ${PROJECT_SOURCE_DIR}/cmake)

add_library(
  ${COMPONENT_NAME} ${CMAKE_CURRENT_BINARY_DIR}/version.c
                    include/xir/xrt_device_handle.hpp src/xrt_device_handle.cpp)

if(XRT_FOUND)
  set(CMAKE_REQUIRED_INCLUDES "${XRT_INCLUDE_DIRS}")
  set(CMAKE_REQUIRED_LIBRARIES "${XRT_CORE_LIBRARIES}")
  include(CheckCXXSourceCompiles)
  if(MSVC)
    # linking error on windows
    set(HAS_xclIPName2Index_3 FALSE)
    set(HAS_xclIPName2Index_2 TRUE)
  else(MSVC)
    check_cxx_source_compiles(
      "#include <xrt.h>
 #include <experimental/xrt-next.h>
 int main(int argc, char * argv[]) {return xclIPName2Index(NULL, NULL, NULL);}
"
      HAS_xclIPName2Index_3)
    check_cxx_source_compiles(
      "#include <xrt.h>
 #include <experimental/xrt-next.h>
 int main(int argc, char * argv[]) {return xclIPName2Index(NULL, NULL);}
"
      HAS_xclIPName2Index_2)
  endif(MSVC)
  set(HAS_xclIPName2Index FALSE)
  if(HAS_xclIPName2Index_3)
    set(HAS_xclIPName2Index TRUE)
    target_compile_definitions(${COMPONENT_NAME}
                               PRIVATE -DHAS_xclIPName2Index_3)
  endif(HAS_xclIPName2Index_3)
  if(HAS_xclIPName2Index_2)
    set(HAS_xclIPName2Index TRUE)
    target_compile_definitions(${COMPONENT_NAME}
                               PRIVATE -DHAS_xclIPName2Index_2)
  endif(HAS_xclIPName2Index_2)
  if(NOT HAS_xclIPName2Index)
    message(FATAL_ERROR "cannot find xclIPName2Index")
  endif(NOT HAS_xclIPName2Index)
  target_sources(${COMPONENT_NAME} PRIVATE ${XCLBIN_SRCS})
endif(XRT_FOUND)

if(XRT_FOUND)
  target_sources(${COMPONENT_NAME} PRIVATE src/xrt_bin_stream.hpp
                                           src/xrt_bin_stream.cpp)
endif(XRT_FOUND)

# if(butler_FOUND) target_sources (${COMPONENT_NAME} PRIVATE
# src/xrt_device_handle_butler.hpp src/xrt_device_handle_butler.cpp )
# target_link_libraries (${COMPONENT_NAME} PRIVATE butler::butler)
# endif(butler_FOUND)
if(XRT_FOUND)
  target_sources(${COMPONENT_NAME} PRIVATE src/xrt_device_handle_imp.hpp
                                           src/xrt_device_handle_imp.cpp)
endif(XRT_FOUND)

set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME}
             BUILD_RPATH "\$ORIGIN/../util")
target_link_libraries(
  ${COMPONENT_NAME}
  PRIVATE ${CMAKE_THREAD_LIBS_INIT} Boost::filesystem
  PUBLIC glog::glog util)
if(XRT_FOUND)
  target_link_libraries(${COMPONENT_NAME} PRIVATE XRT::xrt_core
                                                  XRT::xrt_coreutil)
endif(XRT_FOUND)

target_include_directories(
  ${COMPONENT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT dpu
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT dpu
  DESTINATION share/cmake/${PROJECT_NAME})

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/xir/xrt_device_handle.hpp
  COMPONENT dpu
  DESTINATION include/xir)

if(XRT_FOUND)
  add_executable(xrt_read_register test/read_counter.cpp)
  target_link_libraries(xrt_read_register ${COMPONENT_NAME} XRT::xrt_core
                        XRT::xrt_coreutil glog::glog)
  install(
    TARGETS xrt_read_register
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
endif(XRT_FOUND)

if(IS_EDGE)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/vart.conf DESTINATION ../etc)
  install(
    FILES test/reg_edge.conf
    DESTINATION share/vart
    RENAME reg.conf)
else()
  install(
    FILES test/reg_cloud.conf
    DESTINATION share/vart
    RENAME reg.conf)
endif()

if(XRT_FOUND)
  add_executable(test_xrt_device_handle test/test_xrt_device_handle.cpp)
  target_link_libraries(test_xrt_device_handle ${COMPONENT_NAME} XRT::xrt_core
                        XRT::xrt_coreutil glog::glog)
  install(TARGETS test_xrt_device_handle DESTINATION bin)
  if(BUILD_TEST)

    # add_executable(write_counter test/write_counter.cpp)
    # target_link_libraries(write_counter ${COMPONENT_NAME} XRT::xrt_core
    # XRT::xrt_coreutil glog::glog)

    add_executable(write_counter test/write_counter.cpp)
    target_link_libraries(write_counter ${COMPONENT_NAME} XRT::xrt_core
                          XRT::xrt_coreutil glog::glog)

    if(NOT MSVC)
    add_executable(test_ecmd test/test_ecmd.cpp)
    target_link_libraries(test_ecmd ${COMPONENT_NAME} XRT::xrt_core
                          XRT::xrt_coreutil -lpthread)
    endif(NOT MSVC)
    add_executable(test_bin_stream test/test_bin_stream.cpp)
    target_link_libraries(test_bin_stream ${COMPONENT_NAME} XRT::xrt_core
                          XRT::xrt_coreutil glog::glog)
  endif()
  add_executable(test_alloc_bo test/test_alloc_bo.cpp)
  target_link_libraries(test_alloc_bo ${COMPONENT_NAME} XRT::xrt_core
                        XRT::xrt_coreutil glog::glog)
endif()
