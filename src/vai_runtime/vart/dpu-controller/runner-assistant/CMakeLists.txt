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
add_library(
  ${COMPONENT_NAME}
  include/vart/assistant/batch_tensor_buffer.hpp
  src/batch_tensor_buffer.cpp
  include/vart/assistant/tensor_buffer_allocator.hpp
  src/tensor_buffer_allocator.cpp
  src/tensor_buffer_imp_host.cpp
  src/tensor_buffer_imp_host.hpp
  src/tensor_buffer_imp_host_phy.cpp
  src/tensor_buffer_imp_host_phy.hpp
  src/tensor_buffer_imp_view.cpp
  src/tensor_buffer_imp_view.hpp
  src/tensor_buffer_allocator_imp.cpp
  src/tensor_buffer_allocator_imp.hpp)

if(XRT_FOUND)
  target_compile_definitions(${COMPONENT_NAME} PRIVATE -DXRT_FOUND)
endif(XRT_FOUND)

target_sources(
  ${COMPONENT_NAME} PRIVATE include/vart/assistant/xrt_bo_tensor_buffer.hpp
                            src/xrt_bo_tensor_buffer.cpp)
if(XRT_FOUND)
  target_link_libraries(${COMPONENT_NAME} PRIVATE XRT::xrt_core
                                                  XRT::xrt_coreutil)
endif(XRT_FOUND)

add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
target_link_libraries(
  ${COMPONENT_NAME}
  PUBLIC ${PROJECT_NAME}::runner
  PRIVATE ${PROJECT_NAME}::buffer-object ${PROJECT_NAME}::util)

target_include_directories(
  ${COMPONENT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)
set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME})
install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT dpu
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(
  FILES include/vart/assistant/batch_tensor_buffer.hpp
        include/vart/assistant/tensor_buffer_allocator.hpp
        include/vart/assistant/xrt_bo_tensor_buffer.hpp
  DESTINATION include/vart/assistant/)

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT dpu
  DESTINATION share/cmake/${PROJECT_NAME})

add_executable(test_batch_tensor_buffer test/test_batch_tensor_buffer.cpp)
target_link_libraries(test_batch_tensor_buffer)

if(XRT_FOUND)
  add_executable(test_xrt_bo_tensor_buffer test/test_xrt_bo_tensor_buffer.cpp)
  target_link_libraries(test_xrt_bo_tensor_buffer ${COMPONENT_NAME}
                        XRT::xrt_core XRT::xrt_coreutil ${PROJECT_NAME}::util)
endif(XRT_FOUND)

add_executable(test_allocator test/test_allocator.cpp)
target_link_libraries(test_allocator ${COMPONENT_NAME} ${PROJECT_NAME}::util
                      ${CMAKE_THREAD_LIBS_INIT})
