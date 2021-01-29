<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Library v1.3</h1>
    </td>
 </tr>
 </table>

# Introduction
The Vitis AI Library is a set of high-level libraries and APIs built for efficient AI inference with Deep-Learning Processor Unit (DPU). It is built based on the Vitis AI Runtime with Unified APIs, and it fully supports XRT 2020.2.

The Vitis AI Library provides an easy-to-use and unified interface by encapsulating many efficient and high-quality neural networks. This simplifies the use of deep-learning neural networks, even for users without knowledge of deep-learning or FPGAs. The Vitis AI Library allows users to focus more on the development of their applications, rather than the underlying hardware.

For edge users, click 
[Quick Start For Edge](#quick-start-for-edge) to get started quickly. 

For cloud users, click 
[Quick Start For Alveo](#quick-start-for-alveo) to get started quickly.

## Directory Structure Introduction

```
Vitis_AI_Library
├── apps
│   ├── seg_and_pose_detect
│   └── segs_and_roadline_detect
└── samples
    ├── 3Dsegmentation
    ├── classification
    ├── covid19segmentation
    ├── dpu_task
    ├── facedetect	
    ├── facefeature
    ├── facelandmark
    ├── facequality5pt
    ├── hourglass
    ├── lanedetect
    ├── medicaldetection
    ├── medicalsegcell
    ├── medicalsegmentation
    ├── multitask
    ├── openpose
    ├── platedetect
    ├── platenum
    ├── pointpillars
    ├── posedetect
    ├── refinedet
    ├── reid
    ├── retinaface
    ├── segmentation
    ├── ssd
    ├── tfssd
    ├── yolov2
    ├── yolov3	
    └── yolov4
```

## Quick Start For Edge
### Setting Up the Host
Follow steps 1-5 in [Setting Up the Host](../../tools/Vitis-AI-Library/README.md#setting-up-the-host) to set up the host for edge.

### Setting Up the Target
Follow steps 1-4 in [Setting Up the Target](../../tools/Vitis-AI-Library/README.md#setting-up-the-target) to set up the target.
	 	  
### Running Vitis AI Library Examples
Follow [Running Vitis AI Library Examples](../../tools/Vitis-AI-Library/README.md#running-vitis-ai-library-examples) to run Vitis AI Library examples.

## Quick Start For Alveo
### Setting Up the Host for U50/U50lv/U280
Follow steps 1-3 in [Setting Up the Host](../../tools/Vitis-AI-Library/README.md#setting-up-the-host-for-u50u50lvu280) to set up the host for Alveo.

### Running Vitis AI Library Examples for U50/U50lv/U280
Follow [Running Vitis AI Library Examples](../../tools/Vitis-AI-Library/README.md#running-vitis-ai-library-examples-for-u50u50lvu280) to run Vitis AI Library examples.

### Setting Up the Host for U200/U250

For setting up the host for U200/U250 refer to [README](../../tools/Vitis-AI-Library/README.md#setting-up-the-host-for-alveo-u200alveo-u250).

### Running Vitis AI Library Examples for U200/U250

Demo samples are not supported for U200/U250. To run Vitis AI Library examples for U200/U250 refer to [README](../../tools/Vitis-AI-Library/README.md#running-vitis-ai-library-examples-on-alveo-u200alveo-u250-with-dpucadx8g).

## Reference
For more information, please refer to [vitis-ai-library-user-guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_3/ug1354-xilinx-ai-sdk.pdf).
