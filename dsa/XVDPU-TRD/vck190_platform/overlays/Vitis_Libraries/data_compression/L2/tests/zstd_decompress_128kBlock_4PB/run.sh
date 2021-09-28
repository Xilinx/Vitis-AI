#!/bin/bash
EXE_FILE=$1
LIB_PROJ_ROOT=$2
XCLBIN_FILE=$3
echo "XCL_MODE=${XCL_EMULATION_MODE}"
if [ "${XCL_EMULATION_MODE}" != "hw_emu" ] 
then
    cp $LIB_PROJ_ROOT/common/data/sample.txt .
  
    echo -e "\n\n----------Comparing files after Decompression---------\n"
    cmd1=$(diff data/sample.txt data/sample.txt.zst.orig)
    if [ $? -eq 0 ]
     then
        echo "PASS: Files are the same"
    else
        echo "ERROR: Files are different"
        echo "$cmd1"
   fi     
fi
