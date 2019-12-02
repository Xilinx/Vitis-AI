#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -O2 -I=/usr/include/drm -o segs_and_roadline_detect_drm segs_and_roadline_detect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lglog -ldpmultitask  -ldproadline -ldrm -lpthread -DUSE_DRM=1
$CXX -std=c++11 -O2 -I=/usr/include/drm -o segs_and_roadline_detect_x segs_and_roadline_detect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lglog -ldpmultitask  -ldproadline -ldrm -lpthread -DUSE_DRM=0
