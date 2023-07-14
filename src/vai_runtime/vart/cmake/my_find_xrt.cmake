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
unset(XRT_INCLUDE_DIRS)
unset(XRT_CORE)
unset(XRT_COREUTIL)
if(NOT IS_EDGE)
  find_path(
    XRT_INCLUDE_DIRS
    NAMES xrt.h
    PATHS "/opt/xilinx/xrt/include"
    PATH_SUFFIXES xrt)

  find_library(
    XRT_CORE
    NAMES xrt_core
    HINTS "/opt/xilinx/xrt/lib")
  find_library(
    XRT_COREUTIL
    NAMES xrt_coreutil
    HINTS "/opt/xilinx/xrt/lib")
else()
  find_path(
    XRT_INCLUDE_DIRS
    NAMES xrt.h
    PATH_SUFFIXES xrt)
  find_library(XRT_CORE NAMES xrt_core)
  find_library(XRT_COREUTIL NAMES xrt_coreutil)
endif(NOT IS_EDGE)
set(XRT_VERSION "2.3.1301")
mark_as_advanced(
  XRT_FOUND
  XRT_CLOUD_FOUND
  XRT_EDGE_FOUND
  XRT_CORE
  XRT_COREUTIL
  XRT_INCLUDE_DIRS
  XRT_VERSION)
include(FindPackageHandleStandardArgs)
if(XRT_CORE
   AND XRT_COREUTIL
   AND XRT_INCLUDE_DIRS)
  message(
    STATUS
      "XRT FOUND: XRT_LIBRARYIES=${XRT_LIBRARYIES} XRT_INCLUDE_DIRS=${XRT_INCLUDE_DIRS}"
  )
  set(XRT_FOUND TRUE)
else()
  message(
    WARN
    "XRT FOUND: XRT_LIBRARYIES=${XRT_LIBRARYIES} XRT_INCLUDE_DIRS=${XRT_INCLUDE_DIRS}"
  )
  set(XRT_FOUND FALSE)
endif()

if(XRT_FOUND)
  add_library(XRT::xrt_core SHARED IMPORTED)
  set_property(TARGET XRT::xrt_core PROPERTY IMPORTED_LOCATION ${XRT_CORE})
  set_property(
    TARGET XRT::xrt_core
    APPEND
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${XRT_INCLUDE_DIRS})
  get_filename_component(XRT_LIB_DIRECTORY ${XRT_CORE} DIRECTORY)
  set_property(
    TARGET XRT::xrt_core
    APPEND
    PROPERTY INTERFACE_LINK_DIRECTORIES ${XRT_LIB_DIRECTORY})
  set(XRT_CORE_LIBRARIES ${XRT_CORE}) # xrt-device-handle/CMakeLists.txt
                                      # requires this.

  add_library(XRT::xrt_coreutil SHARED IMPORTED)
  set_property(TARGET XRT::xrt_coreutil PROPERTY IMPORTED_LOCATION
                                                 ${XRT_COREUTIL})
  set_property(
    TARGET XRT::xrt_coreutil
    APPEND
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${XRT_INCLUDE_DIRS})
  get_filename_component(XRT_LIB_DIRECTORY ${XRT_COREUTIL} DIRECTORY)
  set_property(
    TARGET XRT::xrt_coreutil
    APPEND
    PROPERTY INTERFACE_LINK_DIRECTORIES ${XRT_LIB_DIRECTORY})
endif()

if(IS_EDGE)
  set(XRT_EDGE_FOUND true)
  set(XRT_CLOUD_FOUND false)
else()
  set(XRT_EDGE_FOUND false)
  set(XRT_CLOUD_FOUND true)
endif()
