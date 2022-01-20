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

INPUT_PB=quantized_model/quantize_eval_model.pb
INPUT_HEIGHT=224
INPUT_WIDTH=224
INPUT_NODE=input
OUTPUT_NODE=resnet_v1_50/predictions/Reshape_1
LABEL_OFFSET=1
IMAGE_DIR=data/Imagenet/val_dataset
IMAGE_LIST=data/Imagenet/val.txt
EVAL_BATCH=8

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
    EVAL_ITER=50000 
    THREADS=4    
else
    EVAL_ITER=2640
    THREADS=5
fi

export XLNX_BUFFER_POOL=16

python code/tf_resnet50_inference.py \
    --input_graph $INPUT_PB                        \
    --eval_image_path $IMAGE_DIR                   \
    --eval_image_list $IMAGE_LIST                  \
    --input_node $INPUT_NODE                       \
    --output_node $OUTPUT_NODE                     \
    --input_height $INPUT_HEIGHT                   \
    --input_width $INPUT_WIDTH                     \
    --label_offset $LABEL_OFFSET                   \
    --preprocess_type vgg                          \
    --eval_batch $EVAL_BATCH                       \
    --nthreads $THREADS                             \
    --eval_iter $EVAL_ITER                         \
    --mode $MODE

export XLNX_BUFFER_POOL=0
