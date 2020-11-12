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

echo " =====> Begin testing quantized model....."
python code/test.py  --caffepath $CAFFE_PATH --modelpath ./quantized/ --prototxt_file quantize_test.prototxt --weight_file fix_train_test.caffemodel --imgpath ./data/EDD/images --savepath ./quant_results/ --num-classes 2 

echo " =====> moU evaluation ..."
python code/eval_edd.py --GT_maskDIR ./data/EDD/labels/ --Eval_maskDIR ./quant_results/ --Img_DIR ./quant_results/BE/ 



