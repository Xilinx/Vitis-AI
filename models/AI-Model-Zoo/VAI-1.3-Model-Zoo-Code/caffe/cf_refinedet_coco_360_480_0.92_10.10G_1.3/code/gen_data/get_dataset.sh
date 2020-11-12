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
SUB_DIR="coco2014"
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

echo "Download COCO train-val2014 anotation zip file..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
rm -f annotations_trainval2014.zip
cp annotations/instances_train2014.json .
cp annotations/instances_val2014.json .
rm -r annotations 

echo "Download COCO train2014 image zip file..."
wget -c http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm -f train2014.zip
    
echo "Download COCO val2014 image zip file..."
wget -c http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm -f val2014.zip

echo "Done"
  
cd $WORK_DIR


## convert coco style to voc style.

python convert_coco2voc_like.py

## create lmdb for model 

python create_data.py 
