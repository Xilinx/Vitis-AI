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

recipe_path=/tmp/wego_example_recipes/pytorch

#img_url='https://github.com/pytorch/hub/raw/master/images/dog.jpg'
img_url="${recipe_path}/images/dog.jpg"

python run.py                                                   \
        --config_file ./config.yaml                             \
        --serialized_model_path ${serialized_wego_module_path}  \
        --img_url ${img_url}

unset XLNX_BUFFER_POOL
