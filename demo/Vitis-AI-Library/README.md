<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Library v1.4</h1>
    </td>
 </tr>
 </table>

# Introduction
The Vitis AI Library is a set of high-level libraries and APIs built for efficient AI inference with Deep-Learning Processor Unit (DPU). It is built based on the Vitis AI Runtime with Unified APIs, and it fully supports XRT 2020.2.

The Vitis AI Library provides an easy-to-use and unified interface by encapsulating many efficient and high-quality neural networks. This simplifies the use of deep-learning neural networks, even for users without knowledge of deep-learning or FPGAs. The Vitis AI Library allows users to focus more on the development of their applications, rather than the underlying hardware.

For edge users, click 
[Quick Start For Edge](#quick-start-for-edge) to get started quickly. 

For cloud users, click 
[Quick Start For Cloud](#quick-start-for-cloud) to get started quickly.

## Directory Structure Introduction

```
Vitis_AI_Library
├── apps
│   ├── multitask_v3_quad_windows
│   ├── seg_and_pose_detect
│   └── segs_and_roadline_detect
└── samples
    ├── 3Dsegmentation
    ├── bcc
    ├── centerpoint
    ├── classification
    ├── covid19segmentation
    ├── dpu_task
    ├── facedetect
    ├── facefeature
    ├── facelandmark
    ├── facequality5pt
    ├── graph_runner
    ├── hourglass
    ├── lanedetect
    ├── medicaldetection
    ├── medicalsegcell
    ├── medicalsegmentation
    ├── multitask
    ├── multitaskv3
    ├── openpose
    ├── platedetect
    ├── platenum
    ├── pmg
    ├── pointpainting
    ├── pointpillars
    ├── pointpillars_nuscenes
    ├── posedetect
    ├── rcan
    ├── refinedet
    ├── reid
    ├── retinaface
    ├── RGBDsegmentation
    ├── segmentation
    ├── ssd
    ├── tfssd
    ├── yolov2
    ├── yolov3
    └── yolov4

```

## Quick Start For Edge
### Setting Up the Host
For `MPSOC`, follow [Setting Up the Host](../../setup/mpsoc/VART#step1-setup-cross-compiler) to set up the host for edge.  
For `VCK190`, follow [Setting Up the Host](../../setup/vck190#step1-setup-cross-compiler) to set up the host for edge.

### Setting Up the Target
For `MPSOC`, follow [Setting Up the Target](../../setup/mpsoc/VART/README.md#step2-setup-the-target) to set up the target.  
For `VCK190`, follow [Setting Up the Target](../../setup/vck190/README.md#step2-setup-the-target) to set up the target.
	 	  
### Running Vitis AI Library Examples
Follow [Running Vitis AI Library Examples](../../tools/Vitis-AI-Library/README.md#running-vitis-ai-library-examples) to run Vitis AI Library examples.

Note: When you update from VAI1.3 to VAI1.4, refer to the following to modify your compilation options.
1. For Petalinux 2021.1, it uses OpenCV4, and for Petalinux 2020.2, it uses OpenCV3. So set the `OPENCV_FLAGS` as needed. You can refer to the following.
```
result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi
```
2. Include `-lvitis_ai_library-dpu_task` in the build script.


## Quick Start For Cloud
### Setting Up the Host for U50/U50lv/U280
Follow [Setting Up the Host](../../tools/Vitis-AI-Library/README.md#setting-up-the-host-for-u50u50lvu280) to set up the host for U50/U50lv/U280.

### Setting Up the Host for VCK5000
Follow [Setting Up the Host](../../setup/vck5000) to set up the host for VCK5000.

### Running Vitis AI Library Examples for U50/U50lv/U280/VCK5000
Follow [Running Vitis AI Library Examples](../../tools/Vitis-AI-Library/README.md#running-vitis-ai-library-examples-for-u50u50lvu280vck5000) to run Vitis AI Library examples.

### Setting Up the Host for U200/U250

For setting up the host for U200/U250 refer to [README](../../tools/Vitis-AI-Library/README.md#setting-up-the-host-for-alveo-u200alveo-u250).

### Running Vitis AI Library Examples for U200/U250

Demo samples are not supported for U200/U250. To run Vitis AI Library examples for U200/U250 refer to [README](../../tools/Vitis-AI-Library/README.md#running-vitis-ai-library-examples-on-alveo-u200alveo-u250-with-dpucadx8g).

## Reference
For more information, please refer to [vitis-ai-library-user-guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_4/ug1354-xilinx-ai-sdk.pdf).
