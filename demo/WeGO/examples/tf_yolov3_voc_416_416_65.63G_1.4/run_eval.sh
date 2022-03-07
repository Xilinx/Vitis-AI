# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

WEIGHTS=quantized_model/quantize_eval_model.pb
SUB_DATA=data/voc2007_test
IMG_DIR=${SUB_DATA}/images
TEST_LIST=${SUB_DATA}/test.txt
GT_FILE=${SUB_DATA}/gt_detection.txt
RESULT_FILE=data/dt_detection.txt
THREADS=12
BATCH=8

usage()
{
    echo "Usage: bash run_eval.sh --[accuracy|perf]"
    exit 0
}

if [ "$#" -ne 1 ];then
    usage
fi

while [[ "$#" -gt 0 ]];do
    case $1 in 
        --accuracy|--perf) m=${1}; MODE=${m: 2}; shift;;
        *) echo "[Error] Unknown parameter passed: $1"; usage; exit 1;;
    esac
    shift
done

echo "[Info] Running with '$MODE' mode..."

export XLNX_BUFFER_POOL=16

if [ -f ${RESULT_FILE} ];then
    rm ${RESULT_FILE}	
fi

python code/tf_yolov3_inference.py   \
    --input_graph ${WEIGHTS}         \
    --eval_image_path ${IMG_DIR}     \
    --eval_image_list ${TEST_LIST}   \
    --result_file ${RESULT_FILE}     \
    --nthreads ${THREADS}            \
    --batch ${BATCH}                 \
    --mode ${MODE}                   \

if [ $? -ne 0 ];then 
   echo "[Error] $MODE test failed."
   exit 1
fi

if [ "$MODE" = "accuracy" ]; then
    python code/evaluation.py          \
        -mode detection                \
        -detection_use_07_metric True  \
        -gt_file ${GT_FILE}            \
        -result_file ${RESULT_FILE}    \
        -detection_iou 0.5             \
        -detection_thresh 0.005 | tee -a ${TEST_LOG}
fi

export XLNX_BUFFER_POOL=0
