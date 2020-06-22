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

#!/usr/bin/env bash

checkpoint_dir=path_to_facereid_model
data_dir=path_to_facereid_dataset

python3 test.py --config_file='configs/facereid_small.yml' \
--dataset='facereid' \
--dataset_root=${data_dir}'/face_reid' \
--load_model=${checkpoint_dir}'/facereid_small.pth.tar' \
--gpu=0 \
 | tee ./test_facereid_small.log

