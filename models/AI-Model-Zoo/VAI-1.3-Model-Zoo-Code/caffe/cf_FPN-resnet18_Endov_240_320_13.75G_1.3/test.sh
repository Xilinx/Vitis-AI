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

# prepare dataset and caffe models
# run test'
CAFFE_PATH=
if [ ! -n "$CAFFE_PATH" ]; then
echo "'CAFFE_PATH' is empty!"
echo "Please set 'CAFFE_PATH' correctly"
exit 0
fi

echo "======> Begin testing....."
python code/test/test.py  --caffepath ${CAFFE_PATH} --modelpath ./float/ --prototxt_file Endov_FPN_R18_OS16_LR.prototxt --weight_file Endov_FPN_R18_OS16_LR.caffemodel --imgpath ./data/Endov/val/ --savepath ./results/ --num-classes 3

echo "======> Finish testing."