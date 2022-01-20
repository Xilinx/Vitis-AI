#
# Copyright 2021 Xilinx Inc.
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

cp utils/run_oie.py ./run_oie.py
sudo cp utils/stacked_alternating_lstm_cpu.py $CONDA_PREFIX/lib/python3.*/site-packages/allennlp/modules/stacked_alternating_lstm.py
python run_oie.py --in=./test/test_in.txt --out=./output/result_test.txt --model-path=./weights/ --batch-size=1 --cuda-device=-1
