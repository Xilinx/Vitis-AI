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

# PART OF THIS FILE AT ALL TIMES.

#bash
GEN_DATA="../../data/train"
TRAIN_IMG_DIR=$GEN_DATA/WIDER_train/images/
ORIG_TRAIN_ANNO_FILE=$GEN_DATA/WIDER_train/wider_face_train_bbx_gt.txt
OUT_TRAIN_ANNO_FILE=$GEN_DATA/wider_list.txt

CAFFE_ROOT="../../../../../caffe-xilinx"
LMDB_PATH=$GEN_DATA/widerface_train_lmdb


WORK_DIR=$(pwd)"/"

echo $WORK_DIR
if [ -d $GEN_DATA ] && [ -d $TRAIN_IMG_DIR ] && [ -f $ORIG_TRAIN_ANNO_FILE ]
then echo "WIDER_train and ANNO exist"
else
    echo "Start generating train-data list"
    python gen_traindata_list.py --inputAnnoFile $ORIG_TRAIN_ANNO_FILE --outputAnnoFile $OUT_TRAIN_ANNO_FILE
    echo "Complete list generation"
fi

echo "Start generating lmdb"

if [ -d $LMDB_PATH ] 

then  echo "lmdb already exists in this directory. If you want to regenerate, delete the current lmdb file"
    
else
$CAFFE_ROOT/build/tools/convert_face $TRAIN_IMG_DIR $OUT_TRAIN_ANNO_FILE $LMDB_PATH
fi
echo "Complete lmdb generation"
