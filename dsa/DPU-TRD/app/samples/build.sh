result=0 && pkg-config --list-all | grep opencv4 && result=1
if [  -eq 1 ]; then
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi

$CXX -std=c++17 -Llib -Iinclude -Isrc src/resnet50.cpp -lglog -lvart-mem-manager -lxir -lunilog -lvart-buffer-object -lvart-runner -lvart-util -lvart-xrt-device-handle  -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lvart-dpu-runner -lvart-dpu-controller
