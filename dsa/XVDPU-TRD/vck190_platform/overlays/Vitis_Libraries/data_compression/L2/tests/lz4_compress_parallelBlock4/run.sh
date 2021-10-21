#!/bin/bash
EXE_FILE=$1
LIB_PROJ_ROOT=$2
XCLBIN_FILE=$3
echo "XCL_MODE=${XCL_EMULATION_MODE}"
if [ "${XCL_EMULATION_MODE}" != "hw_emu" ]
then
    cp $LIB_PROJ_ROOT/common/data/sample.txt ./sample_run.txt
    for ((i = 0 ; i < 10 ; i++))
    do
        find ./reports/ -type f | xargs cat >> ./sample_run.txt
    done

    echo -e "\n\n-----------Running Compression for large file-----------\n"
    cmd1="$EXE_FILE -c ./sample_run.txt -xbin $XCLBIN_FILE"
    echo $cmd1
    $cmd1
fi
