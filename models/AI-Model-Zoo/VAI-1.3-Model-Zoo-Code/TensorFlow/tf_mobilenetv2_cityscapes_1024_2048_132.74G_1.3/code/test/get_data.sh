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

#!/bin/bash

# usage:
#  bash data/get_data.sh
DATA_ROOT="data"
DATA_NAME="cityscapes"
DATA_DIR="leftImg8bit"
ANNO_DIR="gtFine"
WORK_DIR=$(pwd)"/"

echo " =====> Processing data..."
# relabel annotation *_labelId.png to [0, 18] + [255]
python code/test/utils/process_data.py  --root $DATA_ROOT/$DATA_NAME \
              --images_dir $DATA_DIR  \
              --annotations_dir $ANNO_DIR \
              --image_suffix  '_leftImg8bit.png' \
              --annotation_suffix '_gtFine_trainIds.png' \

