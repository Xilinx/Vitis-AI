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

cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1
CXX=${CXX:-g++}
os=`lsb_release -a | grep "Distributor ID" | sed 's/^.*:\s*//'`
os_version=`lsb_release -a | grep "Release" | sed 's/^.*:\s*//'`
arch=`uname -p`
target_info=${os}.${os_version}.${arch}
install_prefix_default=$HOME/.local/${target_info}

result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi

name=$(basename $PWD)
if [[ "$CXX"  == *"sysroot"* ]];then
$CXX -O2 -fno-inline -I. \
     -I=/usr/include/opencv4 \
     -I=/install/Debug/include \
     -I=/install/Release/include \
     -L=/install/Debug/lib \
     -L=/install/Release/lib \
     -I$PWD/../common  -o $name -std=c++17 \
     $PWD/src/main.cc \
     $PWD/../common/common.cpp  \
     -Wl,-rpath=$PWD/lib \
     -lvart-runner \
     ${OPENCV_FLAGS} \
     -lopencv_videoio  \
     -lopencv_imgcodecs \
     -lopencv_highgui \
     -lopencv_imgproc \
     -lopencv_core \
     -lglog \
     -lxir \
     -lunilog \
     -lpthread
else
$CXX -O2 -fno-inline -I. \
     -I${install_prefix_default}.Debug/include \
     -I${install_prefix_default}.Release/include \
     -L${install_prefix_default}.Debug/lib \
     -L${install_prefix_default}.Release/lib \
     -Wl,-rpath=${install_prefix_default}.Debug/lib \
     -Wl,-rpath=${install_prefix_default}.Release/lib \
     -I$PWD/../common  -o $name -std=c++17 \
     $PWD/src/main.cc \
     $PWD/../common/common.cpp  \
     -Wl,-rpath=$PWD/lib \
     -lvart-runner \
     ${OPENCV_FLAGS} \
     -lopencv_videoio  \
     -lopencv_imgcodecs \
     -lopencv_highgui \
     -lopencv_imgproc \
     -lopencv_core \
     -lglog \
     -lxir \
     -lunilog \
     -lpthread
fi
