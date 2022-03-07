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


DATA=data
SUB_DATA=${DATA}/voc2007_test

if [ ! -d ${SUB_DATA} ]
then
    mkdir ${SUB_DATA}
fi

CUR_DIR=$(pwd)
echo "Entering ${DATA}..."
cd ${DATA}

echo "Do you want to download VOC2007 test dataset?"
read -p "Enter 'Y' or 'y' to download, else use the data you have prepared in ${DATA}: " choice
case ${choice} in
Y | y)
    echo "Prepare to download VOC2007 test dataset..."
    wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar;;
*)
    echo "Using data in ${DATA} you have prepared following the instruction in README.md";;
esac

echo "Entering ${CUR_DIR}"
cd ${CUR_DIR}

SRC_LIST=${DATA}/VOCdevkit/VOC2007/ImageSets/Main/test.txt
DST_LIST=${SUB_DATA}/test.txt
cp ${SRC_LIST} ${DST_LIST}

SRC_IMG_DIR=${DATA}/VOCdevkit/VOC2007/JPEGImages
DST_IMG_DIR=${SUB_DATA}/images
cp -r ${SRC_IMG_DIR} ${DST_IMG_DIR}

ANNO_DIR=${DATA}/VOCdevkit/VOC2007/Annotations
DST_ANNO=${SUB_DATA}/gt_detection.txt
python code/dataset_tools/convert_voc_anno.py -anno_dir ${ANNO_DIR} -list_file ${SRC_LIST} -dst_file ${DST_ANNO}

echo "Finished!"
