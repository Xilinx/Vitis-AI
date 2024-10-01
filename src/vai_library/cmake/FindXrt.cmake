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
if(NOT CMAKE_CROSSCOMPILING)
  find_path(XRT_INCLUDE_DIRS
    NAMES xrt.h
    PATHS "/opt/xilinx/xrt/include"
    PATH_SUFFIXES xrt
    )

  find_library(XRT_LIBRARYIES
    NAMES xrt_core
    HINTS "/opt/xilinx/xrt/lib"
    )
else()
  find_path(XRT_INCLUDE_DIRS
    NAMES xrt.h
    PATH_SUFFIXES xrt
    )

  find_library(XRT_LIBRARYIES
    NAMES xrt_core
    )
endif(NOT CMAKE_CROSSCOMPILING)
set(XRT_VERSION "2.3.1301")
mark_as_advanced(XRT_FOUND XRT_CLOUD_FOUND XRT_EDGE_FOUND XRT_LIBRARYIES XRT_INCLUDE_DIRS XRT_VERSION)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Xrt
    REQUIRED_VARS XRT_INCLUDE_DIRS XRT_LIBRARYIES
    VERSION_VAR XRT_VERSION
)

if(XRT_FOUND AND NOT TARGET XRT::XRT)
  add_library(XRT::XRT SHARED IMPORTED)
  set_property(TARGET XRT::XRT PROPERTY IMPORTED_LOCATION  ${XRT_LIBRARYIES})
  set_property(TARGET XRT::XRT APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${XRT_INCLUDE_DIRS})
  get_filename_component(XRT_LIB_DIRECTORY ${XRT_LIBRARYIES} DIRECTORY)
  set_property(TARGET XRT::XRT APPEND PROPERTY INTERFACE_LINK_DIRECTORIES ${XRT_LIB_DIRECTORY})
  set_property(TARGET XRT::XRT APPEND PROPERTY INTERFACE_LINK_LIBRARIES -lxrt_coreutil)
endif()

if(CMAKE_CROSSCOMPILING)
  set(XRT_EDGE_FOUND true)
  set(XRT_CLOUD_FOUND false)
else()
  set(XRT_EDGE_FOUND false)
  set(XRT_CLOUD_FOUND true)
endif()
