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
# bash

## configure environment
#bash

GEN_DATA_DIR="../../data/"
SUB_DIR="VOC"
WORK_DIR=$(pwd)"/"

if [ ! -d $GEN_DATA_DIR ]
then
    mkdir $GEN_DATA_DIR
    mkdir $GEN_DATA_DIR$SUB_DIR

fi

if [ ! -d $GEN_DATA_DIR$SUB_DIR ]
then
    mkdir $GEN_DATA_DIR$SUB_DIR 
fi
   
cd $GEN_DATA_DIR$SUB_DIR

echo "Download ...."

echo "Download VOC test2007 image zip file..."

wget -c http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar

SRC_IMG_DIR=VOCdevkit/VOC2007/JPEGImages
DST_IMG_DIR=images
cp -r ${SRC_IMG_DIR} ${DST_IMG_DIR}

rm -r VOCdevkit
rm -f VOCtest_06-Nov-2007.tar
    
echo "Done"
  
cd $WORK_DIR


