#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_multitask test_accuracy_multitask.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpmultitask  -ldpproto -ldpmath -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_jpeg_multitask test_jpeg_multitask.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpmultitask  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_multitask test_performance_multitask.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpmultitask  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_video_multitask test_video_multitask.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpmultitask  -pthread -lglog
