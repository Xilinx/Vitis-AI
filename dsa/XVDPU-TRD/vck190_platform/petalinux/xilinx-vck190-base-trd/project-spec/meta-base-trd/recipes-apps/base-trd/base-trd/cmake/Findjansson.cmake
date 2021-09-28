#
# Copyright 2021 Xilinx Inc.
# SPDX-License-Identifier: Apache-2.0
#

#
# Findjansson
# ----------
# Finds the jansson util library
#
# This will define the following variables:
#
# JANSSON_FOUND - system has jansson
# JANSSON_INCLUDE_DIRS - the jansson include directory
# JANSSON_LIBRARIES - the jansson libraries
#

find_path(JANSSON_INCLUDE_DIRS NAMES jansson.h)
find_library(JANSSON_LIBRARIES NAMES jansson)

set (_JANSSON_REQUIRED_VARS JANSSON_LIBRARIES JANSSON_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(jansson REQUIRED_VARS ${_JANSSON_REQUIRED_VARS})

mark_as_advanced(JANSSON_INCLUDE_DIRS JANSSON_LIBRARIES)
