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

#!/bin/sh
Dataset_path='../../../../data/dataset/'
Results='./../../../results/'
Float_model='../../../../float/models/'
Split='valid'
Network='salsanext'
Cuda='0'
Dump='True'
export CUDA_VISIBLE_DEVICES=0
export W_QUANT=1

cd ./train/tasks/semantic/; 
./infer.py --nndct_quant_opt=3 -d $Dataset_path -l $Results -m $Float_model -n $Network -s $Split -C $Cuda -mode 'float'
echo "finishing infering.\n Starting evaluating"
./evaluate_iou.py -d $Dataset_path -p $Results --split $Split -m $Float_model
