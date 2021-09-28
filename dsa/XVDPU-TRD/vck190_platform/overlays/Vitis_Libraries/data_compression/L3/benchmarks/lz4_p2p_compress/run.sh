#!/bin/bash
EXE_FILE=$1
LIB_PROJ_ROOT=$2
XCLBIN_FILE=$3
echo "XCL_MODE=${XCL_EMULATION_MODE}"
if [ "${XCL_EMULATION_MODE}" != "hw_emu" ] 
then
    echo -e "\n\n-----------Running Compression for file in list mode-----------\n"
    cmd1="$EXE_FILE -l ./test.list -cx $XCLBIN_FILE"
    echo $cmd1
    $cmd1
fi
