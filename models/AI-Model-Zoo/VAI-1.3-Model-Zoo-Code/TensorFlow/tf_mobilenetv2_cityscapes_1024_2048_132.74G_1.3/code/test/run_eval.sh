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


DATA_NAME=cityscapes
DATASET_PATH=data
WEIGHTS_PATH=float
MODEL_NAME=Deeplabv3_plus

echo "======================================================================================="
echo ">>>>>>>Step1: Generate gray prediction"
GPU_ID=0
SAVE_DIR=results
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --data_folder $DATASET_PATH/$DATA_NAME/leftImg8bit/val/ --pb_file $WEIGHTS_PATH/$MODEL_NAME/final_model_1024x2048_0514.pb --nclass 19 --target_h 1024 --target_w 2048 --savedir ${SAVE_DIR}

echo ">>>>>>>Step2: Evaluate mIoU"
if [ ! -d "./gtFine_val_gt" ]; then
  mkdir ./gtFine_val_gt/
  cp $DATASET_PATH/$DATA_NAME/gtFine/val/**/*_trainIds.png ./gtFine_val_gt/
fi

python code/test/utils/evaluate_miou.py --task segmentation --gt ./gtFine_val_gt --result ${SAVE_DIR} --result_suffix 'leftImg8bit.png' --num_classes 19 --ignore_label 255 --result_file 'accuracy.txt'
rm -rf ${SAVE_DIR}
rm -rf gtFine_val_gt

echo ">>>>>>>Evaluation finishes!"
echo ">>>>>>>Save performance result in 'accuracy.txt'"
echo "======================================================================================="
