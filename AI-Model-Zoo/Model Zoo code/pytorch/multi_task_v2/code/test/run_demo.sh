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

# PART OF THIS FILE AT ALL TIMES.

#!/bin/bash
# trained_model " PATH OF THE MODEL"
# image_root ="ROOT PATH OF IMAGES"
# image_list = "DETECTIONI IMAGES LIST"
DATASET=../../data
WEIGHTS=../../float/pytorch_multi-task_v2_resnet18_512*320_mod_input.pth
IMG_LIST=demo.txt

python demo.py --trained_model ${WEIGHTS} --image_root ${DATASET} --image_list ${IMG_LIST} 
