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
check_config GT_JSON
check_config DT_JSON
check_config LABELMAP
check_config SCORE_THRESHOLD
check_config GPUS
check_config TEST_SCRIPT

python $TEST_SCRIPT \
    --model $QUANTIZE_EVAL_MODEL \
    --data-root $TEST_IMAGE_DIR \
    --image-list $TEST_IMAGE_LIST \
    --gt-json $GT_JSON \
    --output $DT_JSON \
    --labelmap $LABELMAP \
    --score-threshold $SCORE_THRESHOLD \
    --gpus ${GPUS} \
    --use_quantize True

rm -f $DT_JSON
echo "[INFO]Evaluate quantize model finished"
