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


CAFFE_PATH=
if [ ! -n "$CAFFE_PATH" ]; then
echo "'CAFFE_PATH' is empty!"
echo "Please set 'CAFFE_PATH' correctly"
exit 0
fi
python code/test/test.py  --caffepath ${CAFFE_PATH}  \
                          --modelpath ./float/ \
                          --num-classe 19 \
                          --prototxt_file test.prototxt \
                          --weight_file trainval.caffemodel  \
                          --imgpath ./data/cityscapes/leftImg8bit/val/ \
                          --savepath ./results/ \


echo "======> moU evaluation ..."

if [ ! -d "./gtFine_val_gt" ]; then
  mkdir ./gtFine_val_gt/
fi

cp ./data/cityscapes/gtFine/val/**/*_trainIds.png ./gtFine_val_gt/

python code/utils/evaluate_miou.py --task segmentation  \
                 --gt ./gtFine_val_gt/ \
                 --result ./results/ \
                 --result_suffix 'leftImg8bit.png' \
                 --gt_suffix 'gtFine_trainIds.png' \
                 --num_classes 19 \
                 --ignore_label 255 \
                 --result_file 'accuracy.txt' \

echo "======> Finish moU evaluation." 
echo "Note: If mIoU with float model is 56.69%, testing successfully!"

rm -rf ./gtFine_val_gt/
rm -rf ./results/
