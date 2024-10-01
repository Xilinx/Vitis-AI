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

frozen_model='freeze.pb'

vai_q_tensorflow \
  inspect \
  --input_frozen_graph ${frozen_model} \
  --input_nodes  Input_image_2 \
  --input_shapes ?,128,128,3 \
  --output_nodes output \
  --gpu -1 \
  --target_type DPUCADF8H_ISA0 \
