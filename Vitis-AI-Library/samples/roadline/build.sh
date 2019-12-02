#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_roadline test_accuracy_roadline.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldproadline  -ldpproto -lglog
$CXX -std=c++11 -I. -o test_jpeg_roadline test_jpeg_roadline.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldproadline  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_roadline test_performance_roadline.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldproadline  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_video_roadline test_video_roadline.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldproadline  -pthread -lglog
