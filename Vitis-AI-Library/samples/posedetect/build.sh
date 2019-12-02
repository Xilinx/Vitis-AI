#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_posedetect test_accuracy_posedetect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpposedetect  -ldpproto -lglog
$CXX -std=c++11 -I. -o test_jpeg_posedetect test_jpeg_posedetect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpposedetect  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_jpeg_posedetect_with_ssd test_jpeg_posedetect_with_ssd.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpposedetect  -ldpcommon -pthread -ldpssd -lglog
$CXX -std=c++11 -I. -o test_performance_posedetect test_performance_posedetect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpposedetect  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_posedetect_with_ssd test_performance_posedetect_with_ssd.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpposedetect  -ldpcommon -pthread -ldpssd -lglog
$CXX -std=c++11 -I. -o test_video_posedetect test_video_posedetect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpposedetect  -pthread -lglog
$CXX -std=c++11 -I. -o test_video_posedetect_with_ssd test_video_posedetect_with_ssd.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpposedetect  -pthread -ldpssd -lglog
