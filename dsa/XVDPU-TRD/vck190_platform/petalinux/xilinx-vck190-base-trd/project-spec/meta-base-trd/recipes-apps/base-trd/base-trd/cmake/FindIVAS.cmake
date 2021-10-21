#
# Copyright 2021 Xilinx Inc.
# SPDX-License-Identifier: Apache-2.0
#

#
# FindIVAS
# ----------
# Finds the IVAS util library
#
# This will define the following variables:
#
# IVAS_FOUND - system has IVAS
# IVAS_INCLUDE_DIRS - the IVAS include directory
# IVAS_LIBRARIES - the IVAS libraries
#

find_path(IVAS_INCLUDE_DIRS NAMES ivas/ivas_kernel.h)
find_library(IVAS_LIBRARIES NAMES ivasutil)

set (_IVAS_REQUIRED_VARS IVAS_LIBRARIES IVAS_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IVAS REQUIRED_VARS ${_IVAS_REQUIRED_VARS})

mark_as_advanced(IVAS_INCLUDE_DIRS IVAS_LIBRARIES)
