# Copyright 2019 Xilinx Inc.
#
# Distributed under the Apache License, Version 2.0 (the "License");
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

#[=======================================================================[.rst:
FindXir
-------

Finds the XIR library.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``XIR_FOUND``
True if the system has the XIR library.
``XIR_INCLUDE_DIRS``
Include directories needed to use XIR.
``XIR_LIBRARIES``
Libraries needed to link to XIR.

#]=======================================================================]

find_path(XIR_INCLUDE_DIR NAMES xir/graph/graph.hpp)
find_library(XIR_LIBRARY NAMES xir)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Xir
  FOUND_VAR XIR_FOUND
  REQUIRED_VARS
    XIR_LIBRARY
    XIR_INCLUDE_DIR)

if(XIR_FOUND)
  set(XIR_LIBRARIES ${XIR_LIBRARY})
  set(XIR_INCLUDE_DIRS ${XIR_INCLUDE_DIR})
endif()
