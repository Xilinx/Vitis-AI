#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -O3 -I. -o demo_classification demo_classification.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lglog -ldpbase -ldpproto -lvitis_dpu
