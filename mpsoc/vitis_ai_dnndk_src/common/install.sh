#!/bin/bash

## Copyright 2020 Xilinx Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

echo "Usage 1: ./install                (install on the target board)"
echo "Usage 2: ./install [sysroot_dir]  (install to the sysroot of cross compiler)"
echo ""

install_fail () {
    echo ""
    echo "Installation failed!"
    exit 1
}

SYSROOT=$1
if [ -z "$SYSROOT" ]; then
    echo "Start installing Vitis AI DNNDK on the target board ..."

    [ -f /usr/lib/dpu.xclbin ] || cp /media/sd-mmcblk0p1/dpu.xclbin /usr/lib/dpu.xclbin

    cp -d pkgs/usr/bin/*  /usr/bin/ || install_fail
    cp -d pkgs/usr/lib/*  /usr/lib/ || install_fail
    cp -r pkgs/usr/include/*  /usr/include || install_fail
    ldconfig

    # install python support
    tmp=$(pip3 --version 2>&1 | grep "command not found")
    if [ "$tmp" = "" ]; then
        pip3 install pkgs/python/*.whl || install_fail
    else
        echo "Warning: pip3 command not found, skip install python support"
    fi
else
    echo "Start installing Vitis AI DNNDK to the sysroot of cross compiler ..."
    if [ ! -d "${SYSROOT}/usr/lib" ] || [ ! -d "${SYSROOT}/usr/include" ] || [ ! -d "${SYSROOT}/usr/bin" ]; then
	    echo "Sysroot directory should have usr/lib, usr/include and usr/bin."
	    install_fail
    fi

    cp -d pkgs/usr/bin/*  ${SYSROOT}/usr/bin/ || install_fail
    cp -d pkgs/usr/lib/*  ${SYSROOT}/usr/lib/ || install_fail
    cp -r pkgs/usr/include/*  ${SYSROOT}/usr/include || install_fail
fi


echo ""
echo "Complete installation successfully."

exit 0
