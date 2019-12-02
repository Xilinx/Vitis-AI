#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_yolov2 test_accuracy_yolov2.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpyolov2  -ldpproto -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_jpeg_yolov2 test_jpeg_yolov2.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpyolov2  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_yolov2 test_performance_yolov2.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpyolov2  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_video_yolov2 test_video_yolov2.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpyolov2  -pthread -lglog
