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

#  bash data/get_data.sh

# create dataset soft-link 

GEN_DATA="../data/"
DATA_SET="cityscapes/"
DATASET_DIR="/group/modelzoo/test_dataset/cityscapes_20cls/"
DATA_DIR="leftImg8bit"
ANNO_DIR="gtFine"
LISTS_DIR="lists/"
WORK_DIR=$(pwd)"/"


echo $WORK_DIR
if [ ! -d $GEN_DATA ]
then
  mkdir $GEN_DATA #&& mkdir $GEN_DATA$DATA_SET
else
  echo "../data directory already exists!"
fi


if [ -d $DATASET_DIR ]
then 
  echo " =====> Internal test!"

  if [ -d $DATASET_DIR$ANNO_DIR ] && [ -d $DATASET_DIR$DATA_DIR ] 
  then
    ln -s $DATASET_DIR $GEN_DATA
  else
    echo "=====> Please download datatset!"
    exit 1
  fi
fi


echo " =====> Processing data..."

# relabel annotation *_labelId.png to [0, 18] + [255]
# create train/val lists including [image_path label_path]

if [ ! -d $GEN_DATA$DATA_SET$LISTS_DIR ]; then

  mkdir  $GEN_DATA$DATA_SET$LISTS_DIR
fi

python ./utils/process_data.py  --root $GEN_DATA$DATA_SET \
              --images_dir $DATA_DIR  \
              --annotations_dir $ANNO_DIR \
              --lists_dir $LISTS_DIR \
              --image_suffix  '_leftImg8bit.png' \
              --annotation_suffix '_gtFine_trainids.png' \

DATA_SET_NEW='cityscapes_20cls/'
mkdir -p $GEN_DATA$DATA_SET_NEW
TRAIN_DATA='train_images'
TRAIN_ANNO='train_masks'
VAL_DATA='val_images'
VAL_ANNO='val_masks'
mkdir -p $GEN_DATA$DATA_SET_NEW$TRAIN_DATA
mkdir -p $GEN_DATA$DATA_SET_NEW$TRAIN_ANNO
mkdir -p $GEN_DATA$DATA_SET_NEW$VAL_DATA
mkdir -p $GEN_DATA$DATA_SET_NEW$VAL_ANNO
cd $GEN_DATA
cp -r $DATA_SET$DATA_DIR/train/**/*.png $GEN_DATA$DATA_SET_NEW$TRAIN_DATA
cp -r $DATA_SET$DATA_DIR/val/**/*.png $GEN_DATA$DATA_SET_NEW$VAL_DATA
cp -r $DATA_SET$ANNO_DIR/train/**/*trainids.png $GEN_DATA$DATA_SET_NEW$TRAIN_ANNO
cp -r $DATA_SET$ANNO_DIR/val/**/*trainids.png   $GEN_DATA$DATA_SET_NEW$VAL_ANNO
