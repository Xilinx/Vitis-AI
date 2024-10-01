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
#!/bin/bash

set -e
workspace=$(mktemp -d --dry-run --tmpdir=$CMAKE_CURRENT_BINARY_DIR/tmp)
mkdir -p "$workspace"
cd "$workspace"
mkdir -p samples/lib
echo "this is pwd $PWD, echo $CMAKE_INSTALL_PREFIX"
cp -av                                          \
$CMAKE_INSTALL_PREFIX/lib/libvart-util.so*               \
$CMAKE_INSTALL_PREFIX/lib/libvart-runner.so*                  \
$CMAKE_INSTALL_PREFIX/lib/libvart-runner-assistant.so*        \
$CMAKE_INSTALL_PREFIX/lib/libvart-xrt-device-handle.so*        \
$CMAKE_INSTALL_PREFIX/lib/libvart-buffer-object.so*            \
$CMAKE_INSTALL_PREFIX/lib/libvart-dpu-controller.so*           \
$CMAKE_INSTALL_PREFIX/lib/libvart-dpu-runner.so*               \
$CMAKE_INSTALL_PREFIX/lib/libvart-mem-manager.so*               \
$CMAKE_INSTALL_PREFIX/lib/libvart-trace.so*               \
$CMAKE_INSTALL_PREFIX/lib/libxir.so*               \
$CMAKE_INSTALL_PREFIX/lib/libunilog.so*               \
samples/lib

cp -av $CMAKE_INSTALL_PREFIX/include                  \
samples/
mkdir -p samples/bin
cp -av $CMAKE_CURRENT_BINARY_DIR/resnet50 samples/bin

mkdir -p samples/src
cp -av $CMAKE_CURRENT_SOURCE_DIR/resnet50.cpp samples/src/
cp -av ${CMAKE_CURRENT_BINARY_DIR}/word_list.inc samples/src/
cat <<EOF > samples/build.sh
CXX=\${CXX:-g++}
result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ \$result -eq 1 ]; then
	OPENCV_FLAGS=\$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=\$(pkg-config --cflags --libs-only-L opencv)
fi

\$CXX -std=c++17 -Llib -Iinclude -Isrc src/resnet50.cpp -lglog -lvart-mem-manager -lxir -lunilog -lvart-buffer-object -lvart-runner -lvart-util -lvart-xrt-device-handle \${OPENCV_FLAGS} -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lvart-dpu-runner -lvart-dpu-controller -lvart-runner-assistant -lvart-trace
EOF

tar -zcvf $CMAKE_CURRENT_BINARY_DIR/resnet50.tar.gz samples
echo "CONGRATULATION $CMAKE_CURRENT_BINARY_DIR/resnet50.tar.gz is ready"
trap on_finish EXIT

function on_finish {
    rm -fr "$workspace";
}
