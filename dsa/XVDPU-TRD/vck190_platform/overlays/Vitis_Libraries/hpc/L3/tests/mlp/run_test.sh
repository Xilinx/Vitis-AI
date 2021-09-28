#!/bin/bash

XCLBIN_PATH=$1
MODEL_NUMBER=$2
INPUT_NUMBER=$3

make host OUT_HW_DIR=$1

batch_list="204800"
logs=()

for batch in $batch_list; do
 python data_gen.py --model $2 --batch $batch --inputs $3
 ./build_dir.hw.xilinx_u250_xdma_201830_2/fcn_test.exe $1 data_$batch $2 $batch $3 | tee log-$batch.txt
 logs="$logs log-$batch.txt"
done


egrep -h ^DATA_CSV $logs |  head -1  > perf_QRes_20NN_Single.csv
egrep -h ^DATA_CSV $logs |  grep -v Traces >> perf_QRes_20NN_Single.csv
