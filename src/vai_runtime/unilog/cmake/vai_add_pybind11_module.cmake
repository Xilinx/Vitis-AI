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

option(BUILD_PYTHON "build python interface" OFF)
option(INSTALL_HOME "install python lib in cmake install path" OFF)
option(INSTALL_USER "install python lib in user space" OFF)
if(BUILD_PYTHON)
  if(CMAKE_CROSSCOMPILING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    execute_process(
      COMMAND
        ${Python3_EXECUTABLE} -c
        "from sys import stdout; from distutils import sysconfig; import os;stdout.write(os.path.basename(os.path.dirname(sysconfig.get_python_lib())))"
      OUTPUT_VARIABLE PYTHON_INSTALL_DIR)
    find_path(
      _PYBIND11_PATH pybind11
      HINTS
        /usr/include/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}m
        /usr/include/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})
    get_filename_component(PYBIND11_PATH ${_PYBIND11_PATH} DIRECTORY)
    message(STATUS "PYBIND11_PATH is ${_PYBIND11_PATH}")
    message(
      STATUS "Found Python ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
    if(NOT _PYBIND11_PATH)
      message(WARNING "PYBIND11 NOT FOUND. python extenions are not be built.")
    else(NOT _PYBIND11_PATH)
      get_filename_component(PYBIND11_PATH ${_PYBIND11_PATH} DIRECTORY)
    endif(NOT _PYBIND11_PATH)
    string(REGEX MATCH "python3.[0-9]m" _PYMALLOC "${_PYBIND11_PATH}")
    if(_PYMALLOC)
      set(VAI_PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}m)
    else(_PYMALLOC)
      set(VAI_PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})
    endif(_PYMALLOC)
    set(PYTHON_SITE_PACKAGES "lib/${PYTHON_INSTALL_DIR}/site-packages")
    message("Path for Python Install ${PYTHON_SITE_PACKAGES}")
    set(VAI_PYTHON_LIB python${VAI_PYTHON_VERSION})
    set(VAI_PYTHON_INCLUDE_DIRS
        "${CMAKE_SYSROOT}/usr/include/python${VAI_PYTHON_VERSION}")
  else(CMAKE_CROSSCOMPILING)
    find_package(pybind11 REQUIRED)
    set(PYBIND11_INSTALL OFF)
    set(PYBIND11_TEST OFF)
    set(PYBIND11_PYTHON_VERSION 3)
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-m" "site" "--user-site"
                    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_USER)
    string(REGEX REPLACE "\n" "" PYTHON_SITE_PACKAGES_USER
                         ${PYTHON_SITE_PACKAGES_USER})
  endif(CMAKE_CROSSCOMPILING)
  message(STATUS "Found Python ${PYTHON_EXECUTABLE}")
  message(STATUS "Found INSTALL_HOME=${INSTALL_HOME}")
  message(STATUS "Found INSTALL_USER=${INSTALL_USER}")
  message(STATUS "Found PYTHON_SITE_PACKAGES_USER=${PYTHON_SITE_PACKAGES_USER}")
  message(STATUS "Found PYBIND11_PATH=${PYBIND11_PATH}")
  message(STATUS "Found VAI_PYTHON_LIB=${VAI_PYTHON_LIB}")
  message(STATUS "Found VAI_PYTHON_INCLUDE_DIRS=${VAI_PYTHON_INCLUDE_DIRS}")
endif(BUILD_PYTHON)

function(vai_add_pybind11_module target_name)
  if(NOT BUILD_PYTHON)
    message(FATAL "We must enable BUILD_PYTHON to use vai_add_pybind11_module")
  endif(NOT BUILD_PYTHON)
  cmake_parse_arguments(ARG "" "MODULE_NAME;PACKAGE_NAME" "" ${ARGN})
  if(NOT DEFINED ARG_MODULE_NAME)
    set(ARG_MODULE_NAME ${target_name})
  endif(NOT DEFINED ARG_MODULE_NAME)

  if(NOT DEFINED ARG_PACKAGE_NAME)
    set(ARG_PACKAGE_NAME ".")
  else(NOT DEFINED ARG_PACKAGE_NAME)
    string(REPLACE "." "/" __TMP ${ARG_PACKAGE_NAME})
    set(ARG_PACKAGE_NAME ${__TMP})
  endif(NOT DEFINED ARG_PACKAGE_NAME)

  if(CMAKE_CROSSCOMPILING)
    add_library(${target_name} SHARED ${ARG_UNPARSED_ARGUMENTS})
    set_target_properties(
      ${target_name}
      PROPERTIES
        INCLUDE_DIRECTORIES
        "${CMAKE_CURRENT_SOURCE_DIR}/include;${PYBIND11_PATH};${VAI_PYTHON_INCLUDE_DIRS};${PYTHON_INCLUDE_DIRS}"
        COMPILE_DEFINITIONS "MODULE_NAME=${ARG_MODULE_NAME}"
        PREFIX ""
        LIBRARY_OUTPUT_NAME "${ARG_MODULE_NAME}")
    target_link_libraries(${target_name} PRIVATE -l${VAI_PYTHON_LIB})
  else(CMAKE_CROSSCOMPILING)
    find_package(pybind11 REQUIRED)
    message("target_name is ${target_name}")
    pybind11_add_module(${target_name} SHARED ${ARG_UNPARSED_ARGUMENTS})
    # we need to add ${PYTHON_LIBRARIES}, otherwise linker error because of
    # -Wl,--no-undefined
    target_link_libraries(${target_name} PRIVATE pybind11::module
                                                 ${PYTHON_LIBRARIES})
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME
                                                    "${ARG_MODULE_NAME}")
    set_property(
      TARGET ${target_name}
      APPEND
      PROPERTY COMPILE_DEFINITIONS "MODULE_NAME=${ARG_MODULE_NAME}")
    set_property(
      TARGET ${target_name}
      APPEND
      PROPERTY INCLUDE_DIRECTORIES "${PYTHON_INCLUDE_DIRS}")
  endif(CMAKE_CROSSCOMPILING)
  if(INSTALL_HOME)
    install(TARGETS ${target_name} DESTINATION lib/python/${ARG_PACKAGE_NAME})
  elseif(INSTALL_USER)
    install(TARGETS ${target_name}
            DESTINATION ${PYTHON_SITE_PACKAGES_USER}/${ARG_PACKAGE_NAME})
  else()
    install(TARGETS ${target_name}
            DESTINATION ${PYTHON_SITE_PACKAGES}/${ARG_PACKAGE_NAME})
  endif()
endfunction(vai_add_pybind11_module)
