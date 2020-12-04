#/bin/sh
cd ../../libdpu4rnn
rm -rf build
./make.sh
cp ./build/dpu4rnn_py.so ../app/customer_satisfaction/
