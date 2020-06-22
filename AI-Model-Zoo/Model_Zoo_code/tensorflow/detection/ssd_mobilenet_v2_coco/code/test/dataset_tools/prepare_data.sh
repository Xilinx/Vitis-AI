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

DATA=data
SUB_DATA=${DATA}/coco2014_minival_8059

CUR_DIR=$(pwd)
echo "Entering ${DATA}..."
cd ${DATA}

echo "Prepare to download COCO train-val2014 anotation zip file..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip

echo "Prepare to download COCO val2014 image zip file..."
wget -c http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm -f val2014.zip

echo "Entering ${CUR_DIR}"
cd ${CUR_DIR}

echo "Generating validation list..."
IDX_LIST=code/test/dataset_tools/mscoco_minival_ids.txt
DST_LIST=${SUB_DATA}/minival2014_8059.txt
python code/test/dataset_tools/gen_minival2014_list.py -idx_list ${IDX_LIST} -dst_list ${DST_LIST}

echo "Collecting validation images..."
SRC_DIR=${DATA}/val2014
DST_DIR=${SUB_DATA}/image
if [ ! -d ${DST_DIR} ]
then
    mkdir -p ${DST_DIR}
fi
while IFS= read -r filename
do
    cp ${SRC_DIR}/${filename}.jpg ${DST_DIR}/${filename}.jpg
done < ${DST_LIST}

echo "Generating validation groundtruth json..."
OLD_JSON=${DATA}/annotations/instances_val2014.json
NEW_JSON=${SUB_DATA}/minival2014_8059.json
FILTER_FILE=code/test/dataset_tools/mscoco_minival_ids.txt
python code/test/dataset_tools/gen_minival2014_json.py -old_json_file ${OLD_JSON} -new_json_file ${NEW_JSON} -filter_file ${FILTER_FILE}

echo "Finished!"
