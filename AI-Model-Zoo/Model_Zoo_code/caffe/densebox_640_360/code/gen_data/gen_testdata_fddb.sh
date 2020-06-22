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
GEN_DATA="../../data/test"

FDDB_IMG_DIR=$GEN_DATA/images
FDDB_ANNO_DIR=$GEN_DATA/FDDB-folds
OUT_ANNO_FILE=$GEN_DATA/FDDB_annotations.txt
OUT_IMGLIST_FILE=$GEN_DATA/FDDB_list.txt

if [ -d $FDDB_IMG_DIR ]
then echo "FDDB images directory exists"
else
    mkdir $FDDB_IMG_DIR
fi

WORK_DIR=$(pwd)"/"
echo "$WORK_DIR"
if [ -d $GEN_DATA ]
then echo "testSet path exist"
else
    mkdir $GEN_DATA
fi
cd $GEN_DATA
echo "Downloading FDDB testSet..."

wget -c http://tamaraberg.com/faceDataset/originalPics.tar.gz
wget -c http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz
echo "Unzipping..."  
tar -xf originalPics.tar.gz -C $FDDB_IMG_DIR && rm -r originalPics.tar.gz  
tar -xf FDDB-folds.tgz -C $GEN_DATA && rm -r FDDB-folds.tgz

echo "Done."
cd $WORK_DIR

echo "Start generating FDDB anno list and image list"
python gen_testdata_list.py --oriListPath $FDDB_ANNO_DIR --FDDB_anno $OUT_ANNO_FILE --FDDB_list $OUT_IMGLIST_FILE
echo "Complete list generation"
