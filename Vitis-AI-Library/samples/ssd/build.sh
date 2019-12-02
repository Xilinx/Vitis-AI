#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_accuracy_mlperf_ssd_resnet34_tf test_accuracy_mlperf_ssd_resnet34_tf.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpssd  -ldpproto -ljson-c -ldpmath -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_accuracy_ssd_adas_pruned_0_95 test_accuracy_ssd_adas_pruned_0_95.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpssd  -ldpproto -ldpmath -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_accuracy_ssd_mobilenet_v2 test_accuracy_ssd_mobilenet_v2.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpssd  -ldpproto -ldpmath -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_accuracy_ssd_pedestrain_pruned_0_97 test_accuracy_ssd_pedestrain_pruned_0_97.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpssd  -ldpproto -ldpmath -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_accuracy_ssd_traffic_pruned_0_9 test_accuracy_ssd_traffic_pruned_0_9.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpssd  -ldpproto -ldpmath -lprotobuf -lglog
$CXX -std=c++11 -I. -o test_jpeg_ssd test_jpeg_ssd.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpssd  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_performance_ssd test_performance_ssd.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpssd  -ldpcommon -pthread -lglog
$CXX -std=c++11 -I. -o test_video_ssd test_video_ssd.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpssd  -pthread -lglog
