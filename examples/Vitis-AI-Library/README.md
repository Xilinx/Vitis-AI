<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Library v2.5</h1>
    </td>
 </tr>
 </table>

# Introduction
The Vitis AI Library is a set of high-level libraries and APIs built for efficient AI inference with Deep-Learning Processor Unit (DPU). It is built based on the Vitis AI Runtime with Unified APIs, and it fully supports XRT 2022.1.

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
│   ├── segs_and_roadline_detect
│   ├── vck190_4mipi
│   └── vck190_4video
├── README.md
└── samples
    ├── 3Dsegmentation
    ├── bcc
    ├── c2d2_lite
    ├── centerpoint
    ├── classification
    ├── clocs
    ├── covid19segmentation
    ├── dpu_task
    ├── efficientdet_d2
    ├── facedetect
    ├── facefeature
    ├── facelandmark
    ├── facequality5pt
    ├── fairmot
    ├── graph_runner
    ├── hourglass
    ├── lanedetect
    ├── medicaldetection
    ├── medicalsegcell
    ├── medicalsegmentation
    ├── multitask
    ├── multitaskv3
    ├── ocr
    ├── ofa_yolo
    ├── openpose
    ├── platedetect
    ├── platenum
    ├── pmg
    ├── pointpainting
    ├── pointpillars
    ├── pointpillars_nuscenes
    ├── polypsegmentation
    ├── posedetect
    ├── rcan
    ├── refinedet
    ├── reid
    ├── retinaface
    ├── RGBDsegmentation
    ├── segmentation
    ├── solo
    ├── ssd
    ├── textmountain
    ├── tfssd
    ├── ultrafast
    ├── vehicleclassification
    ├── yolov2
    ├── yolov3
    ├── yolov4
    └── yolovx

```

## Quick Start For Edge
### Setting Up the Host
For `MPSOC`, follow [Setting Up the Host](../../setup/mpsoc#step1-setup-cross-compiler) to set up the host for edge.  
For `VCK190`, follow [Setting Up the Host](../../setup/vck190#step1-setup-cross-compiler) to set up the host for edge.

### Setting Up the Target
For `MPSOC`, follow [Setting Up the Target](../../setup/mpsoc/README.md#step2-setup-the-target) to set up the target.  
For `VCK190`, follow [Setting Up the Target](../../setup/vck190/README.md#step2-setup-the-target) to set up the target.
	 	  
### Running Vitis AI Library Examples
Follow [Running Vitis AI Library Examples](../../src/Vitis-AI-Library/README.md#running-vitis-ai-library-examples) to run Vitis AI Library examples.

Note: When you update from VAI1.3 to VAI2.0 and VAI2.5, refer to the following to modify your compilation options.
1. For Petalinux 2021.1 and above, it uses OpenCV4, and for Petalinux 2020.2, it uses OpenCV3. So set the `OPENCV_FLAGS` as needed. You can refer to the following.
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
### Setting Up the Host

For demonstration purposes, we provide the following pre-compiled DPU IP with Vitis AI Library Sample support. You can choose one of them according to your own Accelerator Card.

| No\. | Accelerator Card | DPU IP |
| ---- | ---- | ----   |
| 1 | U50LV        | DPUCAHX8H         |
| 2 | U50LV        | DPUCAHX8H-DWC     |
| 3 | U55C         | DPUCAHX8H-DWC     |
| 4 | U200         | DPUCADF8H         |
| 5 | U250         | DPUCADF8H         |
| 6 | VCK5000-PROD | DPUCVDX8H_4pe_miscdwc     |
| 7 | VCK5000-PROD | DPUCVDX8H_6pe_dwc  |
| 8 | VCK5000-PROD | DPUCVDX8H_6pe_misc |
| 9 | VCK5000-PROD | DPUCVDX8H_8pe_normal     |

For `U50LV` and `U55C` Alveo Card, follow [Setup Alveo Accelerator Card](../../setup/alveo/README.md) to set up the host.

For `U200` and `U250` Alveo Card, follow [Setup Alveo Accelerator Card](../../setup/alveo/README.md) to set up the host.

For `VCK5000-PROD` Versal Card, follow [Setup VCK5000 Accelerator Card](../../setup/vck5000/README.md) to set up the host.

### Running Vitis AI Library Examples
For `U50LV` and `U55C` Alveo Card, refer to `Running Vitis AI Library Examples on U50LV/U55C/VCK5000` section of [README](../../src/Vitis-AI-Library/README.md#idu50).

For `U200` and `U250` Alveo Card, refer to `Running Vitis AI Library Examples on Alveo-U200/Alveo-U250` section of [README](../../src/Vitis-AI-Library/README.md#idu200).

For `VCK5000-PROD` Versal Card, refer to `Running Vitis AI Library Examples on U50LV/U55C/VCK5000` section of [README](../../src/Vitis-AI-Library/README.md#idu50).

## Reference
For more information, please refer to [vitis-ai-library-user-guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/2_5/ug1354-xilinx-ai-sdk.pdf).
