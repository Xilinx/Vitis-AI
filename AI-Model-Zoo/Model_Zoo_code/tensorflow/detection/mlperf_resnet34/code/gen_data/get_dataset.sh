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

GEN_DATA_DIR="../../data/"
SUB_DIR="coco2017"
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

echo "Download COCO train-val2017 anotation zip file..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm -f annotations_trainval2017.zip
cp annotations/instances_train2017.json .
cp annotations/instances_val2017.json .
rm -r annotations 

echo "Download COCO train2017 image zip file..."
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm -f train2017.zip
    
echo "Download COCO val2017 image zip file..."
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm -f val2017.zip

echo "Done"
  
cd $WORK_DIR


## convert coco style to voc style.

python ../train/dataset/convert_coco2voc_like.py

## create tfrecord for model 

python ../train/dataset/convert_tfrecords.py 
