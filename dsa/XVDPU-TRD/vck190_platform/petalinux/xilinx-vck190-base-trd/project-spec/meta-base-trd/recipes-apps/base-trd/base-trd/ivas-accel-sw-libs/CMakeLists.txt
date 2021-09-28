#
# Copyright 2021 Xilinx Inc.
# SPDX-License-Identifier: Apache-2.0
#

find_package(IVAS REQUIRED)
find_package(jansson REQUIRED)

set(TRD_LIB_INSTALL_DIR "lib/ivas")
set(TRD_IVAS_INSTALL_DIR "share/ivas/${CMAKE_PROJECT_NAME}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -fpermissive -rdynamic -Wl,--no-undefined" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++14")

add_subdirectory(ivas_xfilter2d_pl)
add_subdirectory(ivas_xfilter2d_sw)

