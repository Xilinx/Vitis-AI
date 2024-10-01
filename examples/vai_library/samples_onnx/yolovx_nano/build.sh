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

lib_x="  -lglog -lunilog -lvitis_ai_library-xnnpp -lvitis_ai_library-model_config -lprotobuf -lxrt_core  -lvart-xrt-device-handle  -lvaip-core -lxcompiler-core -lvart-dpu-controller -lxir -lvart-util -ltarget-factory"
lib_onnx=" -lonnxruntime"
lib_opencv=" -lopencv_videoio  -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_core "

name=$(basename $PWD)
if [[ "$CXX"  == *"sysroot"* ]];then
 inc_x="-I=/usr/include/onnxruntime -I=/install/Release/include/onnxruntime -I=/install/Release/include -I=/usr/include/xrt"
 link_x="  -L=/install/Release/lib"
else
 inc_x=" -I/usr/include/onnxruntime  -I/usr/include/xrt"
 link_x="  "
# link_x="  -L/myspace/build/Release/lib"   # test dir; it will be in /usr/lib after image done
fi

 $CXX -O3 -I. \
     ${inc_x} \
     ${link_x}  \
     -o test_${name}_onnx -std=c++17 \
     $PWD/test_${name}_onnx.cpp  \
     ${OPENCV_FLAGS} \
     ${lib_opencv}  \
     ${lib_x}      \
     ${lib_onnx}
 $CXX -O3 -I. \
     ${inc_x} \
     ${link_x}  \
     -o test_accuracy_${name}_onnx -std=c++17 \
     $PWD/test_accuracy_${name}_onnx.cpp  \
     ${OPENCV_FLAGS} \
     ${lib_opencv}  \
     ${lib_x}      \
     ${lib_onnx}
 $CXX -O3 -I. \
     ${inc_x} \
     ${link_x}  \
     -o test_performance_${name}_onnx -std=c++17 \
     $PWD/test_performance_${name}_onnx.cpp  \
     ${OPENCV_FLAGS} \
     ${lib_opencv}  \
     ${lib_x}      \
     ${lib_onnx}


