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

#!/usr/bin/env bash
checkpoint_dir=path_to_personreid_model
dataset_dir=../data
backbone=resnet50
export W_QUANT=0

echo "[Float mode]Evaluating..."
python code/test.py \
--quant_mode=float \
--config_file=code/configs/personreid_${backbone}.yml \
--dataset=market1501 \
--dataset_root=${dataset_dir}/market1501 \
--load_model=${checkpoint_dir}/personreid_${backbone}.pth \
--device=gpu \
--output_path=reid_${backbone}_quant_result \
 | tee ./test_personreid.log


