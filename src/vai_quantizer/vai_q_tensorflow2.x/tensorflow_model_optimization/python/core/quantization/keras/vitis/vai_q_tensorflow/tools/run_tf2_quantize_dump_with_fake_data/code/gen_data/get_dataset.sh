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


#bash
GEN_DATA="../../data/"
DATA_DIR="Imagenet/val_dataset"
ANNO_FILE="Imagenet/val.txt" 
SUB_DIR="Imagenet"
WORK_DIR=$(pwd)"/"
ORI_DATA="validation"

echo $WORK_DIR
if [ -f $GEN_DATA$ANNO_FILE ] && [ -d $GEN_DATA$DATA_DIR ] 
then

  echo "data link is vaild!"

else

  ## Please make sure dataset structure are same as readme said. 

  CUR_GEN_DIR=$GEN_DATA$SUB_DIR
  mkdir $CUR_GEN_DIR

  python gen_data.py --data-dir $GEN_DATA$ORI_DATA --output-dir $GEN_DATA$DATA_DIR --anno-file $GEN_DATA$ANNO_FILE
 
fi

