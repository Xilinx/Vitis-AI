#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_facelandmark test_accuracy_facelandmark.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpfacelandmark  -ldpproto -lglog
$CXX -std=c++11 -I. -o test_jpeg_facelandmark test_jpeg_facelandmark.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpfacelandmark  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_facelandmark test_performance_facelandmark.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpfacelandmark  -ldpcommon -pthread -lglog
