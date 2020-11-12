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
#-gt_file:path of detection groundtruth
#-result_file:path of merged detection result
DATASET=../../../data
#IMG_LIST=demo.txt
GT_FILE=${DATASET}/det_gt.txt
SAVE_FOLDER=../../../result/
DT_FILE=${SAVE_FOLDER}/det_test_all.txt
TEST_LOG=${DATASET}/det_log.txt

python ./evaluate_det.py -gt_file ${GT_FILE} -result_file ${DT_FILE} 
