#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_classification test_accuracy_classification.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpclassification  -ldpproto -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_accuracy_classification_squeezenet test_accuracy_classification_squeezenet.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpclassification  -ldpproto -ldpbase -ldpmath -lglog
$CXX -std=c++11 -I. -o test_jpeg_classification test_jpeg_classification.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpclassification  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_jpeg_classification_squeezenet test_jpeg_classification_squeezenet.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpclassification  -ldpcommon -pthread -ldpbase -ldpmath -lglog
$CXX -std=c++11 -I. -o test_performance_classification test_performance_classification.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpclassification  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_classification_squeezenet test_performance_classification_squeezenet.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpclassification  -ldpcommon -pthread -ldpbase -ldpmath -lglog
$CXX -std=c++11 -I. -o test_video_classification test_video_classification.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpclassification  -pthread -lglog
