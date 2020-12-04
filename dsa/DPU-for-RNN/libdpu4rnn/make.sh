#!/bin/sh
mkdir -p build
cd build
cmake ..
make
cp dpu4rnn_py.so ..
