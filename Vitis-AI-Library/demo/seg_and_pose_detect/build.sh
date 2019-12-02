#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -O2 -I=/usr/include/drm/ -o seg_and_pose_detect_drm seg_and_pose_detect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lglog -ldpmultitask -ldpposedetect -ldpssd -ldrm -lpthread -DUSE_DRM=1
$CXX -std=c++11 -O2 -I=/usr/include/drm/ -o seg_and_pose_detect_x seg_and_pose_detect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lglog -ldpmultitask -ldpposedetect -ldpssd -ldrm -lpthread -DUSE_DRM=0
