# Copyright 2019 Xilinx Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include_directories(${CMAKE_SOURCE_DIR}/src)

set(TARGET_NAME wrapper)
set(MODULE_NAME xir)
set(PYTHON_BUILD_PATH ${CMAKE_BINARY_DIR}/python)

set(MODULE_SRCS wrapper/wrapper.cpp wrapper/pyxir_error_code.cpp)

if(CMAKE_CROSSCOMPILING)
  ## pybind11 does not support cross compiling very well
  find_path(_PYBIND11_PATH pybind11 HINTS /usr/include/python3.7m)
  message("_PYBIND11_PATH is ${_PYBIND11_PATH}")
  if(NOT _PYBIND11_PATH)
    message(WARNING "PYBIND11 NOT FOUND. python extenions for vitis dpu runner will not be built.")
  else()
    find_package(Python3 REQUIRED COMPONENTS Development)
    get_filename_component(PYBIND11_PATH ${_PYBIND11_PATH} DIRECTORY)
    add_library(${TARGET_NAME} SHARED ${MODULE_SRCS})
    set_target_properties(${TARGET_NAME} PROPERTIES
      INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include;${_PYBIND11_PATH};${CMAKE_SYSROOT}/usr/\
include/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}m;${CMAKE_SOURCE_DIR}/include;${CMAKE_SOURCE_DIR}/src"
      COMPILE_DEFINITIONS "MODULE_NAME=${MODULE_NAME}"
      PREFIX ""
      LIBRARY_OUTPUT_NAME ${MODULE_NAME})
    target_link_libraries(${TARGET_NAME} PRIVATE
      -lpython${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}m
      ${PROJECT_NAME})
    set(PYTHON_SITE_PACKAGES_USER "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
    set(PYTHON_SITE_PACKAGES "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
  endif(NOT _PYBIND11_PATH)
else(CMAKE_CROSSCOMPILING)
  find_package(pybind11 REQUIRED)
  set(PYBIND11_CPP_STANDARD -std=c++14)
  set(PYBIND11_INSTALL OFF)
  set(PYBIND11_TEST OFF)
  pybind11_add_module(${TARGET_NAME} MODULE ${MODULE_SRCS})
  set_target_properties(${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_NAME ${MODULE_NAME})
  target_link_libraries(${TARGET_NAME} PRIVATE ${PROJECT_NAME})
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-m" "site" "--user-site"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_USER)
  string(REGEX REPLACE "\n" "" PYTHON_SITE_PACKAGES_USER ${PYTHON_SITE_PACKAGES_USER})
endif(CMAKE_CROSSCOMPILING)

if(INSTALL_HOME)
  install(TARGETS ${TARGET_NAME} DESTINATION lib/python)
elseif(INSTALL_USER)
  install(TARGETS ${TARGET_NAME} DESTINATION ${PYTHON_SITE_PACKAGES_USER})
else()
  install(TARGETS ${TARGET_NAME} DESTINATION ${PYTHON_SITE_PACKAGES})
endif()
