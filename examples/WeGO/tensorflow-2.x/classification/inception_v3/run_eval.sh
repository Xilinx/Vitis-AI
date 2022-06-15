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

INPUT_H5=quantized_model/quantized.h5
IMAGE_DIR=/scratch/data/Imagenet/val_dataset
IMAGE_LIST=/scratch/data/Imagenet/val.txt

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

echo "[Info] Running with '$MODE' mode."

if [ "$MODE" = "accuracy" ]; then
    EVAL_ITER=1
    THREADS=1    
else
    EVAL_ITER=2000
    THREADS=16
fi

export XLNX_BUFFER_POOL=16

python code/inference.py \
    --input_graph $INPUT_H5                       \
    --eval_image_path $IMAGE_DIR                   \
    --eval_image_list $IMAGE_LIST                  \
    --nthreads $THREADS                             \
    --batch_iter $EVAL_ITER                         \
    --mode $MODE

export XLNX_BUFFER_POOL=0
