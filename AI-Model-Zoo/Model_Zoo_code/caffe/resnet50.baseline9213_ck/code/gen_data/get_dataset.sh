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
TRAIN_DATA_DIR="Imagenet/train_resize_256"
TRAIN_ANNO_FILE="Imagenet/train.txt" 
VAL_DATA_DIR="Imagenet/val_resize_256"
VAL_ANNO_FILE="Imagenet/val.txt" 
SUB_DIR="Imagenet"
WORK_DIR=$(pwd)"/"
TRAIN_ORI_DATA="train"
VAL_ORI_DATA="validation"

echo $WORK_DIR
if [ -f $GEN_DATA$TRAIN_ANNO_FILE ] && [ -d $GEN_DATA$TRAIN_DATA_DIR ] && [ -f $GEN_DATA$VAL_ANNO_FILE ] && [ -d $GEN_DATA$VAL_DATA_DIR ] 
then

  echo "data link is vaild!"

else
  mkdir $GEN_DATA
  cd $GEN_DATA
  
  rm -r $TRAIN_DATA_DIR
  rm -r $VAL_DATA_DIR
  rm $TRAIN_ANNO_FILE
  rm $VAL_ANNO_FILE
    
  echo "Downloading..."

  #wget -c http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

  #echo "Unzipping..."

  #tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz

  echo "Done."
  
  cd $WORK_DIR

  CUR_GEN_DIR=$GEN_DATA$SUB_DIR
  mkdir $CUR_GEN_DIR

  python gen_data.py --data-dir $GEN_DATA$TRAIN_ORI_DATA --output-dir $GEN_DATA$TRAIN_DATA_DIR --short-size 256 --anno-file $GEN_DATA$TRAIN_ANNO_FILE
 

  python gen_data.py --data-dir $GEN_DATA$VAL_ORI_DATA --output-dir $GEN_DATA$VAL_DATA_DIR --short-size 256 --anno-file $GEN_DATA$VAL_ANNO_FILE

fi

