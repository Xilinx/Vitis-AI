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

cmake_minimum_required(VERSION 3.5)
project(vaitrace VERSION 2.5.20221)
set(VAITRACE_DEST_PATH bin/xlnx/${PROJECT_NAME})

# Generate Version Flags
execute_process(COMMAND date +%F-%T OUTPUT_VARIABLE VAITRACE_BUILD_DATE)
string(STRIP "${VAITRACE_BUILD_DATE}" VAITRACE_BUILD_DATE)
execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND git rev-parse --short HEAD
  OUTPUT_VARIABLE VAITRACE_GIT_VERSION)
string(STRIP "${VAITRACE_GIT_VERSION}" VAITRACE_GIT_VERSION)
configure_file(vaitrace_version.in
               ${CMAKE_CURRENT_BINARY_DIR}/vaitrace_version.py)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/vaitrace_version.py
        DESTINATION ${VAITRACE_DEST_PATH})

# Build xapm Library
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
  # g++ -shared apm.cpp apm_shell.cpp -fPIC -o libxapm.so -I./include
  set(XAPM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vaitraceTools/xapm)
  include_directories(${XAPM_DIR}/include)

  add_library(xapm SHARED ${XAPM_DIR}/apm.cpp ${XAPM_DIR}/apm_shell.cpp)

  target_link_libraries(xapm pthread)
  set_target_properties(xapm PROPERTIES VERSION "${PROJECT_VERSION}"
                                        SOVERSION "${PROJECT_VERSION_MAJOR}")

  install(TARGETS xapm DESTINATION lib)
endif()

# Build DpuMcParser Library
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
  set(DPU_MC_PATH writer/parser/dpu_mc_parser)
  set(DPU_MC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${DPU_MC_PATH})
  set(DPU_MC_DEST ${VAITRACE_DEST_PATH}/${DPU_MC_PATH})
  include_directories(${DPU_MC_SRC})
  aux_source_directory(${DPU_MC_SRC} DPU_MC_SRC_FILES)
  foreach(SRC IN ITEMS  ${DPU_MC_SRC_FILES})
    get_filename_component(_N ${SRC} NAME_WE)
    add_library(${_N} SHARED ${SRC})
    #set_target_properties(${_N} PROPERTIES VERSION "${PROJECT_VERSION}"
    #                       SOVERSION "${PROJECT_VERSION_MAJOR}")
    install(
      FILES ${CMAKE_CURRENT_BINARY_DIR}/lib${_N}.so
      DESTINATION ${DPU_MC_DEST}
      RENAME ${_N}.so
    )
  endforeach()
endif()

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${VAITRACE_DEST_PATH}
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN __pycache__ EXCLUDE
  PATTERN ".*" EXCLUDE
  PATTERN script EXCLUDE
  PATTERN test_cfg EXCLUDE
  PATTERN time_sync EXCLUDE
  PATTERN build EXCLUDE
  PATTERN vaitrace_py EXCLUDE
  PATTERN vaitraceTools/xapm EXCLUDE
  PATTERN vaitrace.py PERMISSIONS WORLD_EXECUTE WORLD_READ GROUP_EXECUTE
                                  GROUP_READ OWNER_EXECUTE OWNER_READ)

# Create Symbol Link
if(NOT MSVC)
  # MSVC does not support symlink properly.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink ./xlnx/${PROJECT_NAME}/vaitrace.py
    ${CMAKE_CURRENT_BINARY_DIR}/vaitrace)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/vaitrace DESTINATION bin)
endif(NOT MSVC)

# Build Python
if(BUILD_PYTHON)
  set(VAITRACE_PY_INSTALL_DIR ${PYTHON_SITE_PACKAGES})
  install(
    DIRECTORY python/vaitrace_py
    DESTINATION ${VAITRACE_PY_INSTALL_DIR}
    FILES_MATCHING
    PATTERN "*.py")
endif()

set(CPACK_PACKAGE_NAME "vaitrace")
set(CPACK_PACKAGE_VENDOR "Humanity")
set(CPACK_SET_DESTDIR OFF)
set(CPACK_PACKAGE_CONTACT "LiYunzhi<liyunzhi@xilinx.com>")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
set(CPACK_RPM_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
include(CPack)
