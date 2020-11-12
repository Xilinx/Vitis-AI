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


GHTS_PATH=float
MODEL_NAME=Deeplabv3_plus
CUDA_VISIBLE_DEVICES=0 python code/test/test.py --data_folder ./data/demo/ --pb_file $WEIGHTS_PATH/$MODEL_NAME/final_model_1024x2048_0514.pb --nclass 19 --target_h 1024 --target_w 2048 --savedir ./data/demo_results/ --gray2color
