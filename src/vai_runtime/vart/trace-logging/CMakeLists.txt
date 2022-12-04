# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")

add_library(${COMPONENT_NAME} include/vitis/ai/tracelogging.hpp
                              src/tracelogging.cpp)

add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})

set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             LIBRARY_OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME}
             CXX_STANDARD 17)

option(VART_TL_ENABLE "Enable VART Trace Logging module." ON)

target_compile_definitions(
  ${COMPONENT_NAME}
  PUBLIC TL_ENABLE_BUILD_FLAG=$<BOOL:${VART_TL_ENABLE}>
  PRIVATE _CRT_SECURE_NO_WARNINGS)

target_include_directories(
  ${COMPONENT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT trace-logging
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib)

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT trace-logging
  DESTINATION share/cmake/${PROJECT_NAME})

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai/
  DESTINATION include/vart/tracelogging
  COMPONENT trace-logging
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h")

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai
  DESTINATION include/vitis
  COMPONENT trace-logging
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h")

if(BUILD_TEST)
  add_executable(test_tracelogging test/test_tracelogging.cpp)
  target_link_libraries(test_tracelogging ${COMPONENT_NAME})
endif()
