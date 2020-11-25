cp backup/run_oie.py ./run_oie.py
cp backup/stacked_alternating_lstm_dpu_new.py allennlp/modules/stacked_alternating_lstm.py
python run_oie.py --in=./test/test_in.txt --out=./output/result_test.txt --model-path=./weights/ --batch-size=1 --cuda-device=-1
