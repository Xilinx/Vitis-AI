#!/bin/bash

mkdir ./tmp
mv ./vitisai/pyxir/BUILD ./tmp
mv ./vitisai/pyxir/python/libpyxir.so ./tmp
rm -rf ./vitisai/pyxir

cd ./vitisai
git clone https://github.com/Xilinx/pyxir.git ./pyxir
cd -

mv ./tmp/BUILD ./vitisai/pyxir
mv ./tmp/libpyxir.so ./vitisai/pyxir/python
rmdir ./tmp

docker pull tensorflow/tensorflow:latest-devel

docker build -t vitisai_delegate_amd64 .

CONTAINER_ID=$(docker run -d vitisai_delegate_amd64 bash)

docker cp $CONTAINER_ID://tensorflow_src/bazel-bin/tensorflow/lite/delegates/vitisai/libvitisai_delegate.so .

docker rm $CONTAINER_ID
