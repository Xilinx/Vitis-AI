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

export XLNX_BUFFER_POOL=16

usage()
{
    echo "Usage: bash deserialize_run.sh [serialized_wego_module_path]"
    exit 0
}

if [ "$#" -ne 1 ];then
    usage
fi

serialized_wego_module_path=${1}

recipe_path=/tmp/wego_example_recipes/tensorflow-2.x

IMAGE_DIR="${recipe_path}/images/classification"

python inference.py                                                   \
        --serialized_model_path ${serialized_wego_module_path}        \
        --eval_image_path $IMAGE_DIR                                  

unset XLNX_BUFFER_POOL
