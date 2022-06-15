# NMS: Non Maximum Suppression Accelerator

xf_sort_nms_accel.cpp is the NMS accelerator file which consists of following submodules. NMS is used in the SSD_mobilenet application.

- xf::cv::Sort_Multiclass : Sort multi class array
- xf::cv::data_arr_nms : NMS processing on the sorted array


## Step to create xo file
```
source < path-to-Vitis-installation-directory >/settings64.sh
source < path-to-XRT-installation-directory >/setup.sh
export OPENCV_INCLUDE=< path-to-opencv-include-folder >
export OPENCV_LIB=< path-to-opencv-lib-folder >
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:< path-to-opencv-lib-folder >
export DEVICE=< path-to-platform-directory >/< platform >.xpfm
make clean
make xo TARGET=hw
```

