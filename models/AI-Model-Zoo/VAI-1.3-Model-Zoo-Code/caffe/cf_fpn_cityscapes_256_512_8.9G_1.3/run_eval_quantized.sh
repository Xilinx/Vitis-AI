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


#!/bin/bash
# usage:
#  bash auto_test.sh

## prepare dataset

echo "======> Prepare dataset "

cd ./code/
bash get_data.sh
cd ..

echo "======> Begin testing quantized model....."

CAFFE_PATH=
if [ ! -n "$CAFFE_PATH" ]; then
echo "'CAFFE_PATH' is empty!"
echo "Please set 'CAFFE_PATH' correctly"
exit 0
fi
python code/test/test.py  --caffepath ${CAFFE_PATH}  \
                          --modelpath ./quantized/ \
                          --num-classe 19 \
                          --prototxt_file quantize_test.prototxt \
                          --weight_file quantize_train_test.caffemodel  \
                          --imgpath ./data/cityscapes/leftImg8bit/val/ \
                          --savepath ./quant_results/ \


echo "======> moU evaluation quantized model..."

if [ ! -d "./gtFine_val_gt" ]; then
  mkdir ./gtFine_val_gt/
fi

cp ./data/cityscapes/gtFine/val/**/*_trainIds.png ./gtFine_val_gt/

python code/utils/evaluate_miou.py --task segmentation  \
                 --gt ./gtFine_val_gt/ \
                 --result ./quant_results/ \
                 --result_suffix 'leftImg8bit.png' \
                 --gt_suffix 'gtFine_trainIds.png' \
                 --num_classes 19 \
                 --ignore_label 255 \
                 --result_file 'quant_accuracy.txt' \

echo "======> Finish moU evaluation quantized model." 

rm -rf ./gtFine_val_gt/
rm -rf ./quant_results/

