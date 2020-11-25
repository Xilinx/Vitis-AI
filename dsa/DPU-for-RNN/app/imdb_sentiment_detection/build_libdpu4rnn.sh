#!/bin/sh
cd ../../libdpu4rnn
rm -rf build
sh ./make.sh
cp build/dpu4rnn_py.so ../app/imdb_sentiment_detection/
