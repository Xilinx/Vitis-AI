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
get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/VitisVersion.cmake)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")

add_library(
  ${COMPONENT_NAME}
  ${CMAKE_CURRENT_BINARY_DIR}/version.c
  include/vart/util_export.hpp
  include/vart/tensor_mirror_attrs.hpp
  include/vitis/ai/simple_config.hpp
  include/vitis/ai/time_measure.hpp
  include/vitis/ai/dim_calc.hpp
  include/vitis/ai/lock.hpp
  include/vitis/ai/variable_bit.hpp
  include/vitis/ai/util_export.hpp
  include/vitis/ai/erl_msg_box.hpp
  include/vitis/ai/weak.hpp
  include/vitis/ai/with_injection.hpp
  src/error_code.cpp
  src/simple_config.cpp
  src/dim_calc.cpp
  src/util_4bit.cpp
  src/xir_helper.cpp
  src/zero_copy_helper.cpp
  src/time_measure.cpp
  src/lock.cpp
  src/file_lock.hpp
  src/variable_bit.cpp
  src/tensor_mirror_attrs.cpp
  include/vitis/ai/plugin.hpp
  src/plugin.cpp
  src/getenv.cpp
  src/getenv.c)

set_property(TARGET ${COMPONENT_NAME} PROPERTY C_STANDARD 11)

if(hiredis_FOUND)
  target_sources(${COMPONENT_NAME} PRIVATE src/redis_lock.hpp
                                           src/redis_lock.cpp)
endif(hiredis_FOUND)
if(MSVC)
  target_sources(${COMPONENT_NAME} PRIVATE src/file_lock_msvc.cpp)
else(MSVC)

  target_sources(
    ${COMPONENT_NAME}
    PRIVATE src/file_lock_lnx.cpp
            # thread pool does not work on MSVC because the bug
            include/vitis/ai/thread_pool.hpp src/thread_pool.cpp)
endif(MSVC)
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})
if(BUILD_SHARED_LIBS)
  target_compile_definitions(${COMPONENT_NAME} PUBLIC -DVART_UTIL_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(${COMPONENT_NAME} PUBLIC -DVART_UTIL_USE_DLL=0)
endif(BUILD_SHARED_LIBS)

set_target_properties(
  ${COMPONENT_NAME}
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME})
target_include_directories(
  ${COMPONENT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

if(hiredis_FOUND)
  target_link_libraries(${COMPONENT_NAME} PRIVATE hiredis::hiredis)
  target_compile_definitions(${COMPONENT_NAME} PRIVATE ENABLE_REDIS)
endif(hiredis_FOUND)

target_link_libraries(${COMPONENT_NAME} PUBLIC xir::xir glog::glog
                                               ${CMAKE_THREAD_LIBS_INIT})
if(MSVC)

else(MSVC)
  target_link_libraries(${COMPONENT_NAME} PUBLIC ${CMAKE_DL_LIBS})
endif(MSVC)
target_compile_definitions(${COMPONENT_NAME} PRIVATE VART_UTIL_EXPORT)
install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  COMPONENT base
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(
  DIRECTORY include/vitis include/vart
  COMPONENT base
  DESTINATION include
  PATTERN "*.cpp" EXCLUDE)
install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT base
  DESTINATION share/cmake/${PROJECT_NAME})

if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
  if(NOT MSVC)
    set(TOOL_NAME "vart_version")
    add_executable(${TOOL_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/vart_version.c)
    target_link_libraries(${TOOL_NAME} dl)
    install(TARGETS ${TOOL_NAME} DESTINATION bin)
  endif(NOT MSVC)
endif()

if(BUILD_TEST)
  add_executable(test_simple_config test/test_simple_config.cpp)
  target_link_libraries(test_simple_config ${COMPONENT_NAME})

  add_executable(test_env_config test/test_env_config.cpp)
  target_link_libraries(test_env_config ${COMPONENT_NAME})

  add_library(
    test_injector_lib test/device_scheduler_imp.cpp test/device_scheduler.cpp
                      test/device_scheduler.hpp)
  target_link_libraries(test_injector_lib glog::glog ${COMPONENT_NAME})
  target_compile_definitions(test_injector_lib PRIVATE test_injector_EXPORT)
  add_executable(test_injector test/test_injector.cpp)
  target_link_libraries(test_injector test_injector_lib)

  add_executable(test_weak test/test_weak.cpp)
  target_link_libraries(test_weak ${COMPONENT_NAME})

  add_executable(test_dim_calc test/test_dim_calc.cpp)
  target_link_libraries(test_dim_calc ${COMPONENT_NAME})

  add_executable(test_lock test/test_lock.cpp)
  target_link_libraries(test_lock ${COMPONENT_NAME})

  add_executable(test_variable_bit test/test_variable_bit.cpp)
  target_link_libraries(test_variable_bit ${COMPONENT_NAME})

  add_executable(test_erl_msg_box test/test_erl_msg_box.cpp)
  target_link_libraries(test_erl_msg_box ${COMPONENT_NAME})

  if(NOT MSVC)
    add_executable(test_thread_pool test/test_thread_pool.cpp)
    target_link_libraries(test_thread_pool ${COMPONENT_NAME})
  endif(NOT MSVC)
  add_executable(test_zero_copy_helper test/test_zero_copy_helper.cpp)
  target_link_libraries(test_zero_copy_helper ${COMPONENT_NAME} xir::xir)
endif()
