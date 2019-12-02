#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_reid test_accuracy_reid.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpreid  -ldpproto -lglog
$CXX -std=c++11 -I. -o test_jpeg_reid test_jpeg_reid.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpreid  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_reid test_performance_reid.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpreid  -ldpcommon -pthread -lglog
