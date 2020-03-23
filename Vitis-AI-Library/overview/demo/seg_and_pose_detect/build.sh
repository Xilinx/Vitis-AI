#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
CXX=${CXX:-g++}
$CXX -std=c++11 -O2 -I=/usr/include/drm/ -o seg_and_pose_detect_drm seg_and_pose_detect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lglog -lvitis_ai_library-multitask -lvitis_ai_library-posedetect -lvitis_ai_library-ssd -ldrm -lpthread -DUSE_DRM=1
$CXX -std=c++11 -O2 -I=/usr/include/drm/ -o seg_and_pose_detect_x seg_and_pose_detect.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lglog -lvitis_ai_library-multitask -lvitis_ai_library-posedetect -lvitis_ai_library-ssd -ldrm -lpthread -DUSE_DRM=0
