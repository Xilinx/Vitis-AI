#!/bin/sh
cp backup/run_oie.py ./run_oie.py
cp backup/stacked_alternating_lstm_dpu.py allennlp/modules/stacked_alternating_lstm.py
python run_oie.py --in=./test/test.oie.sent --out=./output/output_dpu.txt --model-path=./weights/ --batch-size=1 --cuda-device=-1
python ./oie-benchmark/moveConf.py --in=./output/output_dpu.txt --out=./output/output_dpu_tab.txt
python ./oie-benchmark/benchmark.py --gold=./oie-benchmark/oie_corpus/test.oie --out=./output/results.dat --tabbed=./output/output_dpu_tab.txt
