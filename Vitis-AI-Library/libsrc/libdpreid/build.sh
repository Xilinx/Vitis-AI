#!/bin/bash

if test -d "$OECORE_TARGET_SYSROOT"; then
    cmake  --verbose=1 -DCMAKE_TOOLCHAIN_FILE=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake -DXILINX_HEADER_DIR=$OECORE_TARGET_SYSROOT/usr/include/xilinx -DCMAKE_INSTALL_PREFIX=$OECORE_TARGET_SYSROOT/usr -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=on .
else
    cmake  --verbose=1 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=on .
fi

make
