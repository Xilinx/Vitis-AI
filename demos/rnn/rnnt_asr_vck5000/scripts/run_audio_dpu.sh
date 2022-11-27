#!/bin/sh
#
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
#
python inference.py --model_toml configs/rnnt_longdemo.toml --ckpt my_work_dir/best_lstm1616else816.pt --dataset_dir my_work_dir --val_manifest my_work_dir/librivox.json --batch_size 1 --mode 2
