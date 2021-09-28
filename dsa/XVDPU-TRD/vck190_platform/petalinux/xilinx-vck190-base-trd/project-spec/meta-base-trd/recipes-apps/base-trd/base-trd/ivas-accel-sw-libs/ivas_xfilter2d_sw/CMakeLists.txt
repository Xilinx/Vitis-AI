#
# Copyright 2021 Xilinx Inc.
# SPDX-License-Identifier: Apache-2.0
#

set(KERNEL ivas_xfilter2d_sw)

find_package(OpenCV REQUIRED core imgproc)

add_library(${KERNEL} SHARED src/ivas_xfilter2d_sw.cpp)

target_include_directories(${KERNEL} PRIVATE
    ${JANSSON_INCLUDE_DIRS}
    ${IVAS_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${KERNEL}
    ${JANSSON_LIBRARIES}
    ${IVAS_LIBRARIES}
    ${OpenCV_LIBS})

install(TARGETS ${KERNEL} DESTINATION ${TRD_LIB_INSTALL_DIR})
install(FILES json_files/kernel_xfilter2d_sw.json DESTINATION ${TRD_IVAS_INSTALL_DIR})

