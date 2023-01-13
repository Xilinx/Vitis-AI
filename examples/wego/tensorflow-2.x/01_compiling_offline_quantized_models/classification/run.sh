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

IMAGE_DIR=/tmp/wego_example_recipes/tensorflow-2.x/images/classification

usage()
{
    echo "Usage: bash run_eval.sh [inception_v3|mobilenet|mobilenet_v3|resnet50|efficientnet] [normal|perf]"
    exit 0
}

if [ "$#" -ne 2 ];then
    usage
fi

model=${1}
MODE=${2}
INPUT_H5=/tmp/wego_example_recipes/tensorflow-2.x/models/$model/quantized.h5

echo "[Info] Running with '$MODE' mode."

if [ "$MODE" = "accuracy" ]; then
    EVAL_ITER=1
    THREADS=1    
else
    EVAL_ITER=1200
    THREADS=16
fi

export XLNX_BUFFER_POOL=16

python $model/code/inference.py \
    --input_graph $INPUT_H5                       \
    --eval_image_path $IMAGE_DIR                   \
    --nthreads $THREADS                             \
    --batch_iter $EVAL_ITER                         \
    --mode $MODE

export XLNX_BUFFER_POOL=0
