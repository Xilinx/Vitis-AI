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
backbone=resnet18 #or resnet50
export W_QUANT=1

echo "[Calib mode]testing "
python code/test.py \
--quant_mode=calib \
--config_file=code/configs/personreid_${backbone}.yml \
--dataset=market1501 \
--dataset_root=${dataset_dir}/market1501 \
--load_model=${checkpoint_dir}/personreid_${backbone}.pth \
--device=gpu \
--output_path=reid_${backbone} \
 | tee ./test_personreid.log

echo "[Test mode]testing and dumping model"
python code/test.py \
--quant_mode=test \
--config_file=code/configs/personreid_${backbone}.yml \
--dataset=market1501 \
--dataset_root=${dataset_dir}/market1501 \
--load_model=${checkpoint_dir}/personreid_${backbone}.pth \
--device=cpu \
--output_path=reid_${backbone} \
--dump_xmodel \
 | tee ./test_personreid.log


#--load_model=${checkpoint_dir}'/personreid_large.pth.tar' \
