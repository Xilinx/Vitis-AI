#
# Copyright 2021 Xilinx Inc.
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.5)

project(base-trd)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_subdirectory(ivas-accel-sw-libs)

set(TRD_NB_INSTALL_DIR "share/notebooks/${CMAKE_PROJECT_NAME}")
install(DIRECTORY notebooks/ DESTINATION ${TRD_NB_INSTALL_DIR})

