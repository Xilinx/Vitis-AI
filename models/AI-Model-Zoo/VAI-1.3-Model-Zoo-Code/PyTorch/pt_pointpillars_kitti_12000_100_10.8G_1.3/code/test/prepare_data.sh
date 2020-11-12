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


TEST_CODE=${PWD}/code/test
KITTI=data/KITTI
export PYTHONPATH=${TEST_CODE}:${PYTHONPATH}

SPLIT_FILE=${TEST_CODE}/datasets/ImageSets/val.txt
INFO_PKL=${KITTI}/kitti_infos_val.pkl

python ${TEST_CODE}/create_data.py create_kitti_info_file --index_path=${SPLIT_FILE} --data_path=${KITTI} --save_path=${INFO_PKL}
python ${TEST_CODE}/create_data.py create_reduced_point_cloud --data_path=${KITTI} --info_path=${INFO_PKL}

SPLIT_FILE=${TEST_CODE}/datasets/ImageSets/train.txt
INFO_PKL=${KITTI}/kitti_infos_train.pkl

python ${TEST_CODE}/create_data.py create_kitti_info_file --index_path=${SPLIT_FILE} --data_path=${KITTI} --save_path=${INFO_PKL}
python ${TEST_CODE}/create_data.py create_reduced_point_cloud --data_path=${KITTI} --info_path=${INFO_PKL}
python ${TEST_CODE}/create_data.py create_groundtruth_database --data_path=${KITTI}
