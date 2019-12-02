#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_refinedet test_accuracy_refinedet.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldprefinedet  -ldpproto -ldpmath -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_jpeg_refinedet test_jpeg_refinedet.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldprefinedet  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_refinedet test_performance_refinedet.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldprefinedet  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_video_refinedet test_video_refinedet.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldprefinedet  -pthread -lglog
