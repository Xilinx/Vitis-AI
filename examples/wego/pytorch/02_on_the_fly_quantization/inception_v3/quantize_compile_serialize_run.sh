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
    echo "Usage: bash quantize_compile_serialize_run.sh [float_model_path] [calibration_images_folder]"
    exit 0
}

if [ "$#" -ne 2 ];then
    usage
fi

float_model_path=${1}
calibration_images_folder=${2}

recipe_path=/tmp/wego_example_recipes/pytorch

#img_url='https://github.com/pytorch/hub/raw/master/images/dog.jpg'
img_url="${recipe_path}/images/dog.jpg"

python run.py                                                       \
        --config_file ./config.yaml                                 \
        --float_model_path ${float_model_path}                      \
        --calibration_images_folder ${calibration_images_folder}    \
        --calib_set_len 200                                         \
        --serialize                                                 \
        --img_url ${img_url}

unset XLNX_BUFFER_POOL
