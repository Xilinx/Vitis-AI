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

#!/bin/sh
cp utils/run_oie.py ./run_oie.py
sudo cp utils/stacked_alternating_lstm_dpu_v2.py $CONDA_PREFIX/lib/python3.*/site-packages/allennlp/modules/stacked_alternating_lstm.py
export PYTHONPATH=../common:$PYTHONPATH
python run_oie.py --in=./test/test.oie.sent --out=./output/output_dpu.txt --model-path=./weights/ --batch-size=1 --cuda-device=-1
python ./oie-benchmark/moveConf.py --in=./output/output_dpu.txt --out=./output/output_dpu_tab.txt
python ./oie-benchmark/benchmark.py --gold=./oie-benchmark/oie_corpus/test.oie --out=./output/results.dat --tabbed=./output/output_dpu_tab.txt
