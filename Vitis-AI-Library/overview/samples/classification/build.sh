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
$CXX -std=c++11 -I. -o test_accuracy_classification test_accuracy_classification.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-classification  -lvitis_ai_library-model_config -lglog 
$CXX -std=c++11 -I. -o test_accuracy_classification_squeezenet test_accuracy_classification_squeezenet.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-classification  -lvitis_ai_library-model_config -lvitis_ai_library-dpu_task -lvitis_ai_library-math -lglog 
$CXX -std=c++11 -I. -o test_jpeg_classification test_jpeg_classification.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-classification  -pthread -lglog 
$CXX -std=c++11 -I. -o test_jpeg_classification_squeezenet test_jpeg_classification_squeezenet.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-classification  -pthread -lvitis_ai_library-dpu_task -lvitis_ai_library-math -lglog 
$CXX -std=c++11 -I. -o test_performance_classification test_performance_classification.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-classification  -lvart-util -pthread -lglog 
$CXX -std=c++11 -I. -o test_performance_classification_squeezenet test_performance_classification_squeezenet.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-classification  -lvart-util -pthread -lvitis_ai_library-dpu_task -lvitis_ai_library-math -lglog 
$CXX -std=c++11 -I. -o test_video_classification test_video_classification.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-classification  -pthread -lglog 
