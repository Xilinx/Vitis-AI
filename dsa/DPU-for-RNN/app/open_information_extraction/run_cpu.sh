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

#!/bin/sh
cp backup/run_oie.py ./run_oie.py
cp backup/stacked_alternating_lstm_cpu.py allennlp/modules/stacked_alternating_lstm.py
python run_oie.py --in=./test/test.oie.sent --out=./output/output_cpu.txt --model-path=./weights/ --batch-size=1 --cuda-device=-1
python ./oie-benchmark/moveConf.py --in=./output/output_cpu.txt --out=./output/output_cpu_tab.txt
python ./oie-benchmark/benchmark.py --gold=./oie-benchmark/oie_corpus/test.oie --out=./output/results.dat --tabbed=./output/output_cpu_tab.txt
