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


echo "Evaluation mAP"

export PYTHONPATH=${PWD}:${PYTHONPATH}

GPU_ID=0
cd code/test
CUDA_VISIBLE_DEVICES=${GPU_ID} python test.py --model ../../float/refinedet_vgg_9.8G.pb --data-root ../../data/EDD/images --image-list ./dataset_config/val_image_list.txt  --gt_file dataset_config/edd_gt_detection.txt --output ../../data/refinedet_vgg.txt
