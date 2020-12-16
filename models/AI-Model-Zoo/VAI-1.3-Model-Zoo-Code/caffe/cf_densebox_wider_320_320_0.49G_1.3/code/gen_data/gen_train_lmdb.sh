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

convert_face_path="/build/tools/convert_face"
convert_face_path_docker="/bin/convert_face"

caffe_xilinx_dir_docker="/opt/vitis_ai/conda/envs/vitis-ai-caffe/"
caffe_path() {
  exec_name=$1
  exec_path=$CAFFE_ROOT$(eval echo '$'"${exec_name}_path")
  if [ ! -f "$exec_path" ]; then
    echo >&2 "$exec_path does not exist, try use path in pre-build docker"
    exec_path=$caffe_xilinx_dir_docker$(eval echo '$'"${exec_name}_path_docker")
  fi
  echo "$exec_path"
}

caffe_exec() {
  exec_path=$(caffe_path "$1")
  shift
  $exec_path "$@"
}

WORK_DIR=$(pwd)"/"

echo $WORK_DIR
if [ -d $GEN_DATA ] && [ -d $TRAIN_IMG_DIR ] && [ -f $OUT_TRAIN_ANNO_FILE ]
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
caffe_exec convert_face $TRAIN_IMG_DIR $OUT_TRAIN_ANNO_FILE $LMDB_PATH
fi
echo "Complete lmdb generation"
