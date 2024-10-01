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

# 0 skip convert
# 1 convert to float16
# 2 convert to double
# 3 convert to bfloat16
# 4 convert to float32

DTYPE=3
vai_q_tensorflow quantize \
   --input_frozen_graph $FLOAT_MODEL \
   --input_nodes $Q_INPUT_NODE \
   --input_shapes ?,$INPUT_HEIGHT,$INPUT_WIDTH,$CHANNEL_NUM \
   --output_nodes $Q_OUTPUT_NODE \
   --output_dir $QUANTIZE_DIR \
   --convert_datatype $DTYPE \
