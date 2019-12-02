#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_openpose test_accuracy_openpose.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpopenpose  -ldpproto -ljson-c -lglog
$CXX -std=c++11 -I. -o test_jpeg_openpose test_jpeg_openpose.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpopenpose  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_openpose test_performance_openpose.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpopenpose  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_video_openpose test_video_openpose.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpopenpose  -pthread -lglog
