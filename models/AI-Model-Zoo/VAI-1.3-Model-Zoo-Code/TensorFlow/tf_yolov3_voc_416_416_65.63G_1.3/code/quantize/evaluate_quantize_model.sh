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
set -e

source ./config.ini

echo "[INFO]Start evaluate quantize model ..."
check_config QUANTIZE_EVAL_MODEL
check_config TEST_IMAGE_DIR
check_config TEST_IMAGE_LIST
check_config Q_EVAL_INPUT_NODE
check_config Q_EVAL_OUTPUT_NODE
check_config INPUT_HEIGHT
check_config INPUT_WIDTH
check_config RESULT_FILE
check_config GT_FILE
check_config DETECTION_IOU
check_config DETECTION_THRESH


python $INFER_SCRIPT \
    --input_graph $QUANTIZE_EVAL_MODEL \
    --eval_image_path $TEST_IMAGE_DIR \
    --eval_image_list $TEST_IMAGE_LIST \
    --input_node $Q_EVAL_INPUT_NODE \
    --output_node $Q_EVAL_OUTPUT_NODE \
    --input_height $INPUT_HEIGHT \
    --input_width $INPUT_WIDTH \
    --result_file $RESULT_FILE \
    --gpus $GPUS \
    --use_quantize True \

python $TEST_SCRIPT \
    -mode detection \
    -detection_use_07_metric True \
    -gt_file $GT_FILE \
    -result_file $RESULT_FILE \
    -detection_iou $DETECTION_IOU \
    -detection_thresh $DETECTION_THRESH \

rm -f $RESULT_FILE
echo "[INFO]Evaluate quantize model finished"
