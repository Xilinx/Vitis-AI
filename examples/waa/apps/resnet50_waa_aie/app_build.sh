#!/bin/bash

ABS_PATH=$(pwd -P)
FATHER_PATH=$ABS_PATH
XILINX_VITIS_AIETOOLS=$XILINX_VITIS/aietools

cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1
CXX=${CXX:-g++}
os=`lsb_release -a | grep "Distributor ID" | sed 's/^.*:\s*//'`
os_version=`lsb_release -a | grep "Release" | sed 's/^.*:\s*//'`
arch=`uname -p`
target_info=${os}.${os_version}.${arch}
install_prefix_default=$HOME/.local/${target_info}

result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi

mkdir -p bin

PLUGINS_PATH=../../plugins/
PLUGINS_PATH=$(realpath $PLUGINS_PATH)
name=$(basename $PWD).exe
#$CXX -E -CC -P -O2 -w
$CXX -std=c++17 -O3 -w -Wl,--as-needed -o ./bin/$name -Llib -Iinclude -Isrc -Lsrc -I$PLUGINS_PATH/include/aie/ -L$PLUGINS_PATH/include/aie/ -L$PLUGINS_PATH/include/aie/lib/sw/aarch64-linux/ src/main.cc src/common.cpp src/aie_control_xrt.cpp --sysroot=$SYSROOT/ -I$SYSROOT/usr/include/xrt -I$XILINX_VITIS_AIETOOLS/include -D __PS_ENABLE_AIE__ -I$SYSROOT/usr/include/ -I$SYSROOT/usr/include/opencv4 --sysroot=$SYSROOT/ -L$SYSROOT/usr/lib/ -L$XILINX_VITIS_AIETOOLS/lib/aarch64.o -I$CROSS_COMPILE_ENV/usr/include/ -I$CROSS_COMPILE_ENV/usr/include/vart/ -L$CROSS_COMPILE_ENV/usr/lib/ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_calib3d -luuid -lopencv_highgui -lglog -lunilog -lxrt_coreutil -lxrt_core -lxir -lpthread -lxilinxopencl -ladf_api_xrt  -lsmartTilerStitcher -lvart-mem-manager -lvart-buffer-object -lvart-runner -lvart-util -lvart-xrt-device-handle ${OPENCV_FLAGS} -lvart-dpu-runner -lvart-dpu-controller -lvart-runner-assistant -lvart-trace
