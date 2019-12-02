#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_facedetect test_accuracy_facedetect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpfacedetect  -ldpproto -ldpmath -lglog
$CXX -std=c++11 -I. -o test_jpeg_facedetect test_jpeg_facedetect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpfacedetect  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_facedetect test_performance_facedetect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpfacedetect  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_video_facedetect test_video_facedetect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpfacedetect  -pthread -lglog
