#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_segmentation test_accuracy_segmentation.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpsegmentation  -ldpproto -lglog
$CXX -std=c++11 -I. -o test_jpeg_segmentation test_jpeg_segmentation.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpsegmentation  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_segmentation test_performance_segmentation.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpsegmentation  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_video_segmentation test_video_segmentation.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpsegmentation  -pthread -lglog
