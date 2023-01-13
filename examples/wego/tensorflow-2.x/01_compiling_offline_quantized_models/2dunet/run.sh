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

usage()
{
    echo "Usage: bash run_eval.sh [normal|perf]"
    exit 0
}

if [ "$#" -ne 1 ];then
    usage
fi

MODE=${1}
THREAD=8
IMAGE_DIR=/tmp/wego_example_recipes/tensorflow-2.x/images/2dunet
model=/tmp/wego_example_recipes/tensorflow-2.x/models/2dunet/quantized.h5
batch_iter=2000
export PYTHONPATH=${PWD}:${PYTHONPATH}
export XLNX_BUFFER_POOL=16
echo ">>>>>>>>>>>Begin Testing>>>>>>>>>>>>>>>>>>>>>>>>>>>"
CUDA_VISIBLE_DEVICES=0 python code/test.py --input_size 128,128 --img_path $IMAGE_DIR --weight_file $model --num_classes 2 --save_path wego_results_visulization --mode $MODE --thread $THREAD --eval_iter $batch_iter
export XLNX_BUFFER_POOL=0

