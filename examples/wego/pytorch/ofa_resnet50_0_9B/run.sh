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

#bin/bash

# evaluation accuracy
export XLNX_BUFFER_POOL=16

usage()
{
    echo "Usage: bash run.sh [normal|perf]"
    exit 0
}

if [ "$#" -ne 1 ];then
    usage
fi

mode=${1}
recipe_path=/tmp/wego_example_recipes/pytorch

#img_url='http://images.cocodataset.org/test-stuff2017/000000000019.jpg'
img_url="${recipe_path}/images/test.jpg"

python run.py --img_url ${img_url}                                                \
              --model_path ${recipe_path}/models/ofa_resnet50_0_9B/ResNets_int.pt \
              --mode=${mode}                                                      \
              --threads 8                        

unset XLNX_BUFFER_POOL