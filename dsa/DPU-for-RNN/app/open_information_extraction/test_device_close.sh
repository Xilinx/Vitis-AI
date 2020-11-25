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
