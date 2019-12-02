#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_tfssd test_accuracy_tfssd.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldptfssd  -ldpproto -ljson-c -lglog
$CXX -std=c++11 -I. -o test_jpeg_tfssd test_jpeg_tfssd.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldptfssd  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_tfssd test_performance_tfssd.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldptfssd  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_video_tfssd test_video_tfssd.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldptfssd  -pthread -lglog
