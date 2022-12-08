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
usage()
{
    echo "Usage: bash run_eval.sh [normal|perf]"
    exit 0
}

if [ "$#" -ne 1 ];then
    usage
fi

MODE=${1}
THREAD=16
batch_iter=200
SAVE_DIR=results_visulization_erfnet
IMAGE_DIR=/tmp/wego_example_recipes/tensorflow-2.x/images/erfnet
model=/tmp/wego_example_recipes/tensorflow-2.x/models/erfnet/quantized.h5
echo 'perform testing...'

export PYTHONPATH=${PWD}:${PYTHONPATH}

export XLNX_BUFFER_POOL=16

python test_wego.py  --arch cbr --input_size 512,1024 --img_path $IMAGE_DIR --num_classes 20 --weight_file $model --save_path $SAVE_DIR --thread $THREAD --mode $MODE --batch_iter $batch_iter 

export XLNX_BUFFER_POOL=0
