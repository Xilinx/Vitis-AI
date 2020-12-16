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

echo "[INFO]Start Quantize ..."
check_config FLOAT_MODEL
check_config INPUT_HEIGHT
check_config INPUT_WIDTH
check_config PREPROCESS_TYPE
check_config Q_INPUT_NODE
check_config Q_OUTPUT_NODE
check_config METHOD
check_config CALIB_IMAGE_DIR
check_config CALIB_IMAGE_LIST
check_config CALIB_BATCH_SIZE
check_config CALIB_INPUT_FN
check_config CALIB_ITER
check_config QUANTIZE_DIR
check_config GPUS
check_config TEST_SCRIPT

if [ ! -d "$QUANTIZE_DIR" ];then
  mkdir $QUANTIZE_DIR
fi

vai_q_tensorflow quantize \
  --input_frozen_graph $FLOAT_MODEL \
  --input_nodes $Q_INPUT_NODE \
  --input_shapes ?,$INPUT_HEIGHT,$INPUT_WIDTH,3 \
  --output_nodes $Q_OUTPUT_NODE \
  --input_fn $CALIB_INPUT_FN \
  --method $METHOD \
  --gpu $GPUS \
  --calib_iter $CALIB_ITER \
  --output_dir $QUANTIZE_DIR \

echo "[INFO]Quantization finished, results are in $QUANTIZE_DIR"
