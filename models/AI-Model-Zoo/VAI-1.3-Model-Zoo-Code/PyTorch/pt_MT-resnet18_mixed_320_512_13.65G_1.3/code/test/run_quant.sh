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

#DATASET and WEIGHTS softlink will be created automatically

#echo "Creating environment..."
#conda env create -f ./code/configs/environment.yaml
#source activate Multi-task_test_env

echo "Preparing dataset..."
CAL_DATASET=../../data/BDD_Waymo_Cityscape_mixdata/
DATASET=../../data
WEIGHTS=../../float/pytorch_multi-task_v2_resnet18_512*320_mod_input.pth


echo "Conducting calibration test..."
export CUDA_VISIBLE_DEVICES=0
IMG_LIST=mix_data.txt

python -W ignore test.py --img_mode 1 --trained_model ${WEIGHTS} --image_root ${CAL_DATASET} --image_list ${IMG_LIST} --eval --quant_mode calib

echo "Conducting quantized Detection test..."
export CUDA_VISIBLE_DEVICES=0
IMG_LIST=det_val.txt
#IMG_LIST=demo.txt
GT_FILE=${DATASET}/det_gt.txt
SAVE_FOLDER=./result
DT_FILE=${SAVE_FOLDER}/det_test_all.txt
TEST_LOG=${SAVE_FOLDER}/det_log.txt

python -W ignore test.py  --img_mode 2 --trained_model ${WEIGHTS}  --image_root ${DATASET} --image_list ${IMG_LIST} --eval --quant_mode test
cat ./result/det/* > ${DT_FILE}
python evaluation/evaluate_det.py -gt_file ${GT_FILE} -result_file ${DT_FILE} | tee -a ${TEST_LOG}
echo "Test report is saved to ${TEST_LOG}"


echo "Conducting quantized Segmentation test..."
export CUDA_VISIBLE_DEVICES=0
IMG_LIST=seg_val.txt
SAVE_FOLDER=./result
GT_FILE=${DATASET}/seg_label/
DT_FILE=${SAVE_FOLDER}/seg/
TEST_LOG=${SAVE_FOLDER}/seg_log.txt
python -W ignore test.py --img_mode 2 --trained_model ${WEIGHTS}  --image_root ${DATASET} --image_list ${IMG_LIST} --eval --quant_mode test
python evaluation/evaluate_seg.py seg ${GT_FILE} ${DT_FILE} | tee -a ${TEST_LOG}
echo "Test report is saved to ${TEST_LOG}"

python test.py --dump_xmodel --img_mode 2 --eval --trained_model ${WEIGHTS} --image_root ${DATASET} --image_list ${IMG_LIST} --quant_mode test

echo "Dumping xmodel..."
export CUDA_VISIBLE_DEVICES=3
IMG_LIST=det_val.txt
#IMG_LIST=demo.txt
GT_FILE=${DATASET}/det_gt.txt
SAVE_FOLDER=./result
DT_FILE=${SAVE_FOLDER}/det_test_all.txt
TEST_LOG=${SAVE_FOLDER}/det_log.txt

python test.py --dump_xmodel --img_mode 2 --trained_model ${WEIGHTS}  --image_root ${DATASET} --image_list ${IMG_LIST} --eval --quant_mode test
#echo "Conducting Demo visual..."
#IMG_LIST=demo.txt

#python test_code/demo.py --trained_model ${WEIGHTS} --image_root ${DATASET} --image_list ${IMG_LIST}
