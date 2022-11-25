# Copyright 2019 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

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
url="${recipe_path}/images/020738.jpg"

python run.py                                                                \
       --img_url=${url}                                                      \
       --model_path ${recipe_path}/models/face_quality/PointsQuality_int.pt  \
       --mode=${mode}                                                        \
       --threads 16                                

unset XLNX_BUFFER_POOL
