#
# Copyright 2021 Xilinx Inc.
# SPDX-License-Identifier: Apache-2.0
#

set(KERNEL ivas_xfilter2d_pl)

add_library(${KERNEL} SHARED src/ivas_xfilter2d_pl.c)

target_include_directories(${KERNEL} PRIVATE
    ${JANSSON_INCLUDE_DIRS}
    ${IVAS_INCLUDE_DIRS})

target_link_libraries(${KERNEL}
    ${JANSSON_LIBRARIES}
    ${IVAS_LIBRARIES})

install(TARGETS ${KERNEL} DESTINATION ${TRD_LIB_INSTALL_DIR})
install(FILES json_files/kernel_xfilter2d_pl.json DESTINATION ${TRD_IVAS_INSTALL_DIR})

