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
check_config TASK
check_config DATA_FOLDER
check_config SAVE_DIR
check_config RESULT_SUFFIX
check_config GT_FOLDER
check_config GT_SUFFIX
check_config NUM_CLASSES
check_config IGNORE_LABEL
check_config RESULT_FILE
check_config GPUS

python $INFER_SCRIPT \
    --pb_file $QUANTIZE_EVAL_MODEL \
    --data_folder $DATA_FOLDER \
    --savedir ${SAVE_DIR} \
    --target_h $INPUT_HEIGHT \
    --target_w $INPUT_WIDTH \
    --nclass $NUM_CLASSES \
    --gpus $GPUS \
    --use_quantize True



mkdir -p $GT_FOLDER
cp -f ../../data/cityscapes/gtFine/val/**/*_trainIds.png $GT_FOLDER
python $TEST_SCRIPT \
    --task $TASK \
    --gt $GT_FOLDER \
    --gt_suffix $GT_SUFFIX \
    --result $SAVE_DIR \
    --result_suffix $RESULT_SUFFIX \
    --num_classes $NUM_CLASSES \
    --result_file $RESULT_FILE \
    --ignore_label $IGNORE_LABEL

rm -rf $SAVE_DIR $GT_FOLDER
echo "[INFO]Evaluate quantize model finished"
