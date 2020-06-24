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
#variable1 :path of segmentation groundtruth
#variable2 :path of segmentation result

DATASET=../../../data
SAVE_FOLDER=../../../result
GT_FILE=${DATASET}/seg_label/
DT_FILE=${SAVE_FOLDER}/seg/


python evaluate_seg.py seg ${GT_FILE} ${DT_FILE}
