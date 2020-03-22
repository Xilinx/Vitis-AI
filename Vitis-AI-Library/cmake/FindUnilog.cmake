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
FindUnilog
-------

Finds the UNILOG library.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``UNILOG_FOUND``
True if the system has the Unilog library.
``UNILOG_INCLUDE_DIRS``
Include directories needed to use Unilog.
``UNILOG_LIBRARIES``
Libraries needed to link to Unilog.

#]=======================================================================]

find_path(UNILOG_INCLUDE_DIR NAMES UniLog/UniLog.hpp)
find_library(UNILOG_LIBRARY NAMES unilog)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Unilog
  FOUND_VAR UNILOG_FOUND
  REQUIRED_VARS
    UNILOG_LIBRARY
    UNILOG_INCLUDE_DIR)

if(UNILOG_FOUND)
  set(UNILOG_LIBRARIES ${UNILOG_LIBRARY})
  set(UNILOG_INCLUDE_DIRS ${UNILOG_INCLUDE_DIR})
endif()
