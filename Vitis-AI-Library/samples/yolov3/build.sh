#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_yolov3_adas_pruned_0_9 test_accuracy_yolov3_adas_pruned_0_9.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpyolov3  -ldpproto -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_accuracy_yolov3_bdd test_accuracy_yolov3_bdd.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpyolov3  -ldpproto -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_accuracy_yolov3_voc test_accuracy_yolov3_voc.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpyolov3  -ldpproto -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_accuracy_yolov3_voc_tf test_accuracy_yolov3_voc_tf.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpyolov3  -ldpproto -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_jpeg_yolov3 test_jpeg_yolov3.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpyolov3  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_yolov3 test_performance_yolov3.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpyolov3  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_video_yolov3 test_video_yolov3.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpyolov3  -pthread -lglog
