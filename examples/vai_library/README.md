<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# Vitis AI Library Introduction
The Vitis AI Library is a set of high-level libraries and APIs built for efficient AI inference with Deep-Learning Processor Unit (DPU). It is built based on the Vitis AI Runtime with Unified APIs, and it fully supports XRT 2022.2.

The Vitis AI Library provides an easy-to-use and unified interface by encapsulating many efficient and high-quality neural networks. This simplifies the use of deep-learning neural networks, even for users without knowledge of deep-learning or FPGAs. The Vitis AI Library allows users to focus more on the development of their applications, rather than the underlying hardware.

For edge users, click 
[Quick Start For Edge](#quick-start-for-edge) to get started quickly. 

For cloud users, click 
[Quick Start For Cloud](#quick-start-for-cloud) to get started quickly.

## Directory Structure Introduction
```
vai_library
├── apps
│   ├── multitask_v3_quad_windows
│   ├── seg_and_pose_detect
│   ├── segs_and_roadline_detect
│   ├── vck190_4mipi
│   └── vck190_4video
├── README.md
├── samples
│   ├── 3Dsegmentation
│   ├── bcc
│   ├── bevdet
│   ├── c2d2_lite
│   ├── centerpoint
│   ├── cflownet
│   ├── classification
│   ├── clocs
│   ├── covid19segmentation
│   ├── dpu_task
│   ├── efficientdet_d2
│   ├── facedetect
│   ├── facefeature
│   ├── facelandmark
│   ├── facequality5pt
│   ├── fairmot
│   ├── graph_runner
│   ├── hourglass
│   ├── lanedetect
│   ├── medicaldetection
│   ├── medicalsegcell
│   ├── medicalsegmentation
│   ├── monodepth2
│   ├── movenet
│   ├── multitask
│   ├── multitaskv3
│   ├── ocr
│   ├── ofa_yolo
│   ├── openpose
│   ├── platedetect
│   ├── platenum
│   ├── pmg
│   ├── pointpainting
│   ├── pointpillars
│   ├── pointpillars_nuscenes
│   ├── polypsegmentation
│   ├── posedetect
│   ├── rcan
│   ├── refinedet
│   ├── reid
│   ├── retinaface
│   ├── RGBDsegmentation
│   ├── segmentation
│   ├── solo
│   ├── ssd
│   ├── textmountain
│   ├── tfssd
│   ├── ultrafast
│   ├── vehicleclassification
│   ├── yolov2
│   ├── yolov3
│   ├── yolov4
│   ├── yolov5
│   ├── yolov6
│   └── yolovx
└── samples_onnx
    ├── 3DSegmentation
    ├── face_quality
    ├── inception_v3_pt
    ├── movenet
    ├── multitaskv3
    ├── rcan
    ├── reid
    ├── resnet50_pt
    ├── segmentation
    ├── textmountain
    └── vehicle_type_resnet18_pt

```

## Quick Start For Edge
### Setting Up the Host
For `MPSOC`, follow [Setting Up the Host](../../board_setup/mpsoc/board_setup_mpsoc.rst#step-1-setup-cross-compiler) to set up the host for edge.  
For `VCK190`, follow [Setting Up the Host](../../board_setup/vck190/board_setup_vck190.rst#step-1-setup-cross-compiler) to set up the host for edge.

### Setting Up the Target
For `MPSOC`, follow [Setting Up the Target](../../board_setup/mpsoc/board_setup_mpsoc.rst#step-2-setup-the-target) to set up the target.  
For `VCK190`, follow [Setting Up the Target](../../board_setup/vck190/board_setup_vck190.rst#step-2-setup-the-target) to set up the target.
	 	  
### Running vai_library Examples
Follow [Running Vitis AI Library Examples](../../src/vai_library/README.md#running-vitis-ai-library-examples) to run Vitis AI Library examples.

Note: When you update from VAI1.3 to VAI2.0, VAI2.5 or VAI3.0, refer to the following to modify your compilation options.
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

### Running vai_library ONNX Examples
**To improve the user experience, the Vitis AI Runtime packages, ONNX Runtime package, VART samples, Vitis-AI-Library samples and
models have been built into the board image. Therefore, user does not need to install Vitis AI
Runtime packages and model package on the board separately. However, users can still install
the model or Vitis AI Runtime on their own image or on the official image by following these
steps.**

1. Download the ONNX runtime package [vitis_ai_2022.2-r3.0.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2022.2-r3.0.0.tar.gz) and install it on the target board. 
```
tar -xzvf vitis_ai_2022.2-r3.0.0.tar.gz -C /
```
Note: for the official released board image, the VART runtime and ONNX runtime have been pre-installed. 

2. Download the ONNX quantized model package [xilinx_model_zoo-edge-onnx-3.0.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=xilinx_model_zoo-edge-onnx-3.0.0.tar.gz) and install it on the target board.
```
tar -xzvf xilinx_model_zoo-edge-onnx-3.0.0.tar.gz --strip-components 1 -C /
```

3. Enter the directory of samples in the target board. Take `resnet50` as an example.
```
cd ~/Vitis-AI/examples/vai_library/samples_onnx/resnet50_pt
```

4. Run the example.
```
./test_resnet50_pt_onnx /usr/share/vitis_ai_library/models/resnet50_onnx_pt/resnet50_onnx_pt.onnx sample_classification.jpg
```
Note: if there is no executable file `test_resnet50_pt_onnx`, run the `sh build.sh` to build the program.

For the performance and accuracy test, refer to `readme` under examples.


## Quick Start For Cloud
### Setting Up the Host

For demonstration purposes, we provide the following pre-compiled DPU IP with Vitis AI Library Sample support. You can choose one of them according to your own Accelerator Card.

| No\. | Accelerator Card | DPU IP |
| ---- | ---- | ----   |
| 1 | VCK5000-PROD | DPUCVDX8H_4pe_miscdwc     |
| 2 | VCK5000-PROD | DPUCVDX8H_6pe_dwc  |
| 3 | VCK5000-PROD | DPUCVDX8H_6pe_misc |
| 4 | VCK5000-PROD | DPUCVDX8H_8pe_normal     |

For `VCK5000-PROD` Versal Card, follow [Setup VCK5000 Accelerator Card](../../board_setup/vck5000/board_setup_vck5000.rst#setting-up-a-versal-accelerator-card) to set up the host.

### Running Vitis AI Library Examples
Follow [Running Vitis AI Library Examples on VCK5000](../../src/vai_library/README.md#idu50) section to run Vitis AI Library examples.

## Reference
For more information, please refer to [vitis-ai-library-user-guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/2_5/ug1354-xilinx-ai-sdk.pdf).
