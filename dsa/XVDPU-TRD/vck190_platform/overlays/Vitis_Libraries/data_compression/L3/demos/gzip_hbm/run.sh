#!/bin/bash
EXE_FILE=$1
LIB_PROJ_ROOT=$2
BIN_PATH=$3
echo "XCL_MODE=${XCL_EMULATION_MODE}"
export XILINX_LIBZ_XCLBIN=$BIN_PATH
if [ "${XCL_EMULATION_MODE}" == "sw_emu" ]; 
then
    echo -e "\n\n-----------Supported Options-----------\n"
    cmd1="$EXE_FILE -h"
    echo $cmd1
    $cmd1
fi
if [ "${XCL_EMULATION_MODE}" != "hw_emu" ] && [ "${XCL_EMULATION_MODE}" != "sw_emu" ];  
then
    cp $LIB_PROJ_ROOT/common/data/sample.txt sample_run.txt
    cp $LIB_PROJ_ROOT/common/data/test.list test.list
    echo "sample_run.txt.gz" > gzip_test_decomp.list
    echo "sample_run.txt.xz" > zlib_test_decomp.list
    for ((i = 0 ; i < 10 ; i++))
    do
        find ./reports/ -type f | xargs cat >> sample_run.txt
    done
   
    for ((i = 0 ; i < 10 ; i++)) 
    do
	cat sample_run.txt >> sample_run.txt${i}
        echo "sample_run.txt${i}"  >> test.list
        echo "sample_run.txt${i}.gz"  >> gzip_test_decomp.list
        echo "sample_run.txt${i}.xz"  >> zlib_test_decomp.list
    done

echo -e "\n\n-----------ZLIB Flow-----------\n"
    cmd1="$EXE_FILE -t sample.txt -zlib 1"
    echo $cmd1
    $cmd1

echo -e "\n\n-----------GZIP Flow (-xbin option)-----------\n"
    cmd2="$EXE_FILE -xbin $BIN_PATH -t sample.txt"
    echo $cmd2
    $cmd2

echo -e "\n\n-----------GZIP Compress list of files -----------\n"
    cmd2="$EXE_FILE -xbin $BIN_PATH -cfl ./test.list"
    echo $cmd2
    $cmd2

echo -e "\n\n-----------ZLIB Compress list of files -----------\n"
    cmd2="$EXE_FILE -xbin $BIN_PATH -cfl ./test.list -zlib 1"
    echo $cmd2
    $cmd2

echo -e "\n\n-----------GZIP Decompress list of files -----------\n"
    cmd2="$EXE_FILE -xbin $BIN_PATH -dfl ./gzip_test_decomp.list"
    echo $cmd2
    $cmd2

echo -e "\n\n-----------ZLIB Decompress list of files -----------\n"
    cmd2="$EXE_FILE -xbin $BIN_PATH -dfl ./zlib_test_decomp.list"
    echo $cmd2
    $cmd2
fi
