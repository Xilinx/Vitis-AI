#!/bin/bash
EXE_FILE=$1
LIB_PROJ_ROOT=$2
GENPATH=$LD_LIBRARY_PATH
DIR=listdir
export LD_LIBRARY_PATH=$(pwd):$GENPATH
export XILINX_LIBZ_XCLBIN=$3
export XRT_INI_PATH=$(pwd)/xrt.ini
echo "XCL_MODE=${XCL_EMULATION_MODE}"
export XILINX_LIBZ_DEFLATE_BUFFER_SIZE=0
export XILINX_LIBZ_INFLATE_BUFFER_SIZE=0
run_cmd ()
{
    cmd="$EXE_FILE $1"
    echo $cmd
    $cmd
}

if [ -z "$XCL_EMULATION_MODE" ]
then
    cp $LIB_PROJ_ROOT/common/data/sample.txt ./sample_run.txt
    for ((i = 0 ; i < 10 ; i++))
    do
        find ./reports/ -type f | xargs cat >> ./sample_run.txt
    done
   
    echo -e "\n\n-----------Validating a Tiny File-----------------\n" 
    cp $LIB_PROJ_ROOT/common/data/cr_1072987 ./cr_1072987
    run_cmd "-t ./cr_1072987 -cm 0"

    echo -e "\n\n-----------Validating a Large File (Default: compress/uncompress)-----------\n"
    run_cmd "-t ./sample_run.txt -cm 0" 

    echo -e "\n\n-----------Validating a Large File (Chunk Mode: deflate)-----------\n"
    run_cmd "-t ./sample_run.txt -cm 1" 

    echo -e "\n\n-----------Validating Compression Levels (Method: compress2/uncompress)-----------\n\n"
    echo -e "\n\n*********** Z_NO_COMPRESSION ************\n"
    run_cmd "-t ./sample_run.txt -cm 0 -cl 0"
    echo -e "\n\n*********** Z_BEST_SPEED ************\n"
    run_cmd "-t ./sample_run.txt -cm 0 -cl 1"
    echo -e "\n\n*********** Z_BEST_COMPRESSION ************\n"
    run_cmd "-t ./sample_run.txt -cm 0 -cl 9"
    echo -e "\n\n*********** Z_DEFAULT_COMPRESSION ************\n"
    run_cmd "-t ./sample_run.txt -cm 0"

    echo -e "\n\n-----------Validating No Acceleration Mode (-n) -----------\n\n"
    echo -e "\n\n*********** Disable Acceleration ************\n"
    run_cmd "-t ./sample_run.txt -cm 0 -n 0 -cl 9"
    echo -e "\n\n*********** Enable Acceleration ************\n"
    run_cmd "-t ./sample_run.txt -cm 0 -n 1 -cl 9"

    echo -e "\n\n----------- Multi Process Testing -----------\n\n"
    mkdir -p $DIR
    rm -rf listdir.list
    for ((i = 0; i < 20; i++))
    do 
        cp -rf $(pwd)/sample_run.txt $(pwd)/$DIR/file_$i
        echo $(pwd)/$DIR/file_$i >> listdir.list
        echo $(pwd)/$DIR/file_${i}.zlib >> listdir_decomp.list
    done
    
    echo -e "\n\n*********** Compression (cfl) ************\n"
    run_cmd "-cfl listdir.list -v 1 -cm 0" 
    
    echo -e "\n\n*********** DeCompression (dfl) ************\n"
    run_cmd "-dfl listdir_decomp.list -v 1" 
    
    echo -e "\n\n*********** Deflate (cfl) ************\n"
    run_cmd "-cfl listdir.list -v 1 -cm 1"
else 
    echo -e "\n\n-----------Validating Sample.txt-----------------\n" 
    run_cmd "-t ./sample.txt"
fi
