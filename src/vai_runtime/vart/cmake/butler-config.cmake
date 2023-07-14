#
# Copyright 2022-2023 Advanced Micro Devices Inc.
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

find_path(butler_INCLUDE_DIRS
    NAMES butler_client.h
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../XIP/Butler/src/" # for debug
    "${XIP_HOME_DIR}/Butler/src/client" # for deploy
    "$ENV{PREFIX}/include/xip/butler"
)

find_library(butler_LIBRARYIES
    NAMES butler
    PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/../XIP/Butler/src/lib/"
    "${XIP_HOME_DIR}/Butler/src/lib"
    "$ENV{PREFIX}/lib"
)

set(butler_VERSION "1.0.0")
mark_as_advanced(butler_INCLUDE_DIRS butler_VERSION)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(butler
    REQUIRED_VARS butler_INCLUDE_DIRS butler_LIBRARYIES
    VERSION_VAR butler_VERSION
)

if(butler_FOUND AND NOT TARGET butler::butler)
  add_library(butler::butler SHARED IMPORTED)
  set_property(TARGET butler::butler PROPERTY IMPORTED_LOCATION  ${butler_LIBRARYIES})
  set_property(TARGET butler::butler APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${butler_INCLUDE_DIRS})
  get_filename_component(butler_LIB_DIRECTORY ${butler_LIBRARYIES} DIRECTORY)
  set_property(TARGET butler::butler APPEND PROPERTY INTERFACE_LINK_DIRECTORIES ${butler_LIB_DIRECTORY})
  message(STATUS "Butler is found ${butler_LIBRARYIES} ${butler_INCLUDE_DIRS}")
else()
  message(STATUS "Butler is NOT found")
endif()
