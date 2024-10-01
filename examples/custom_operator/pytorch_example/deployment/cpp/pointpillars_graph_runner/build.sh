#
# Copyright 2022-2023 Advanced Micro Devices Inc.
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

result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
        OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
        OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi

EIGEN3_FLAGS=$(pkg-config --cflags --libs-only-L eigen3)

CXX=${CXX:-g++}
$CXX -std=c++17 -O2 -I./src \
        -o sample_pointpillars_graph_runner \
        ./src/anchor.cpp \
        ./src/helper.cpp \
        ./src/main.cpp \
        ./src/parse_display_result.cpp \
        ./src/pointpillars_post.cpp \
        ./src/preprocess.cpp \
        -lglog \
        -lxir \
        -lvart-runner \
        -lvart-util \
        -lvitis_ai_library-graph_runner \
        ${OPENCV_FLAGS} \
        ${EIGEN3_FLAGS} \
        -lopencv_core \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lpthread

