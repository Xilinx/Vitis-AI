# Xilinx Single Source Cosine Similarity Demo

This is a demo of running Graph L3 library. It has two main steps, firstly `Build dynamic library and xclbin`, it may take several hours. Once the xclbin is ready, users should follow steps in the `Run demo`. The demo is set to run in one Alveo U50 board by default. But if users want to use two U50 boards, this can be achieved by simply setting `deviceNeeded` in ../../tests/cosineSimilaritySSDenseIntBench/test_cosineSimilaritySSDense.cpp to 2. 

## Build dynamic library and xclbin
    ./build.sh

## Run demo
    change PROJECTPATH in ../../tests/cosineSimilaritySSDenseIntBench/config.json to graph library's absolute path 
    ./run.sh

