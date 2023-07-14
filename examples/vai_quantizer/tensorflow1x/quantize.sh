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
set -ex

echo "[INFO]Start Quantize ..."
FLOAT_MODEL="float.pb"
INPUT_HEIGHT=28
INPUT_WIDTH=28
CHANNEL_NUM=1
Q_INPUT_NODE="input"
Q_OUTPUT_NODE="dense/BiasAdd"
METHOD=1
CALIB_INPUT_FN=input_fn.calib_input
CALIB_ITER=20
QUANTIZE_DIR="quantized"
GPUS=0

vai_q_tensorflow quantize \
  --input_frozen_graph $FLOAT_MODEL \
  --input_nodes input \
  --input_shapes ?,$INPUT_HEIGHT,$INPUT_WIDTH,$CHANNEL_NUM \
  --output_nodes $Q_OUTPUT_NODE \
  --input_fn $CALIB_INPUT_FN \
  --method $METHOD \
  --gpu $GPUS \
  --calib_iter $CALIB_ITER \
  --output_dir $QUANTIZE_DIR \

echo "[INFO]Quantization finished, results are in $QUANTIZE_DIR"
