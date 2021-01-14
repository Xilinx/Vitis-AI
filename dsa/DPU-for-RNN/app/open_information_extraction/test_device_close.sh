#
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
#

cp backup/run_oie_test.py  run_oie.py
cp backup/stacked_alternating_lstm_dpu.py allennlp/modules/stacked_alternating_lstm.py

flag=xx
n=0
while [ x$flag == xxx ]
do
python run_oie.py --in=./test/test_in.txt --out=./output/result_test1.txt --model-path=./weights/ --batch-size=1 --cuda-device=-1 #2>/dev/null 1>&2
n=$((n+1))
echo "loop counter:$n"
diff output/result_test.txt output/result_test1.txt >/dev/null
if [ $? != 0 ]; then
  flag=ha
  echo "Device REEOR!"
fi
done
