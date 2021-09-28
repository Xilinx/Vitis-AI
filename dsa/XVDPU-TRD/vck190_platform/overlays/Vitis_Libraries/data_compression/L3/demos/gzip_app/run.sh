#!/bin/bash
EXE_FILE=$1
LIB_PROJ_ROOT=$2
XCLBIN_PATH=$3
echo "XCL_MODE=${XCL_EMULATION_MODE}"
export XILINX_LIBZ_XCLBIN="${XCLBIN_PATH}"

echo -e "\n\n-----------ZLIB Flow-----------\n"
    cmd1="$EXE_FILE -xbin $XCLBIN_PATH -t ./sample.txt -zlib 1"
    echo $cmd1
    $cmd1

echo -e "\n\n-----------GZIP Flow-----------\n"
    cmd2="$EXE_FILE -xbin $XCLBIN_PATH -t ./sample.txt"
    echo $cmd2
    $cmd2
