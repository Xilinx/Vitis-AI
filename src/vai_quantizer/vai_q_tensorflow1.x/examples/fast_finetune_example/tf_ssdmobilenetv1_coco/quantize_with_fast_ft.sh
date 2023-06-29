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

# QUANT_DIR is set by run_all script for convenience of testing
if [ -z $QUANT_DIR ]; then
  echo "using $QUANTIZE_DIR as quantize results dir"
else
  export QUANTIZE_DIR=$QUANT_DIR
  echo "using $QUANTIZE_DIR passed by QUANT_DIR as quantize results dir"
fi

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
  --include_fast_ft 1 --fast_ft_mode 1 --fast_ft_epochs 5 \
  --fast_ft_lr 1e-5 --fast_ft_lrcoef 1000

echo "[INFO]Quantization finished, results are in $QUANTIZE_DIR"
