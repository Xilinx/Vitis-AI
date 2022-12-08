# Copyright 2022 Xilinx Inc.
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

model="resnet_v1_50"
mode="normal"

#img_url='https://github.com/pytorch/hub/raw/master/images/dog.jpg'
recipe_path=/tmp/wego_example_recipes/tensorflow-1.x
img_url="${recipe_path}/images/dog.jpg"
model_path="${recipe_path}/models/${model}/quantize_eval_model.pb"

python compile_serialize_run.py  \
        --config_file ./${model}/config.yaml            \
        --model_path ${model_path}                      \
        --mode ${mode}                                  \
        --img_url ${img_url}                                                                     

unset XLNX_BUFFER_POOL
