# WAA-TRD: Whole Application Acceleration - Target Reference Design

## Table of Contents

- [1 Introduction](#1-Introduction)
- [2 Background](#2-Background)
- [3 Directory structure introduction](#3-Directory-structure-introduction)
- [4 WAA-TRD run](#4-WAA-TRD-run)
    - [4.1 Overview](#4.1-Overview)
    - [4.2 Design Files](#42-design-files)
    - [4.3 Build and run the application](#43-build-and-run-the-application)
- [5 Build with new Pre-processing Accelerator](#5-Build-with-new-Pre-processing-Accelerator)    


## 1 Introduction

WAA-TRD demonstrates integration of pre/post processing accelerator with DPU(Deep Neural Network DNN accelerator) for Embedded and cloud platform. The pre/post processing accelerator is implemented using Xilinx® [Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) functions. WAA-TRD provides examples for integrating different pre processing involved in the object classification & detection networks with DPU. 

There are two flows being provided 
1.	Build : Both the pre-processing accelerator and DPU are built from sources. This flow uses DPU-TRD’s make flows.
2.	Pre-built : ML is pre-built and its partial bitfiles are pre-created (using DFx flow). only pre-processing accelerator is built from sources. This flow gives 10x & 5x saving in build time for Embedded platform & cloud platform respectively.

## 2 Background
Input images are preprocessed  before being fed for inference of different deep neural networks. The pre-processing steps vary from network to network. For example, for classification networks like Resnet-50 the input image is resized to 224 x 224 size and then channel-wise mean subtraction is performed before feeding the data to the DNN accelerator. For detection networks like YOLO v3 the input image is resized to 256 x 512 size using letterbox before feeding the data to the DNN accelerator. 


[Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) provides functions optimized for FPGA devices that are drop-in replacements for standard OpenCV library functions. This application demonstrates how Vitis Vision library functions can be used to accelerate pre-processing.

## 3 Directory structure introduction
--------------------------------------------------

```
WAA-TRD
├── README.md
├── accel
│   ├── DPUV3E_3ENGINE                      # DPUv3e Accelerator xo file used in the Pre-built flow
│   ├── DPUV3INT8                           # DPUv3eint8 Accelerator xo file used in the Build flow
│   ├── DPUv2_B4096                         # DPUv2 Accelerator xo file used in the Pre-built flow
│   ├── classification-pre                  # Pre-processing Accelerator for Resnet50 application
│   ├── classification-pre_int8             # Pre-processing Accelerator for Resnet50_int8 application
│   ├── classification-pre_jpeg             # Pre-processing Accelerator for Resnet50_jpeg application
│   ├── detection-pre                       # Pre-processing Accelerator for Adas detection application
│   └── jpeg_decoder                        # JPEG decoder Accelerator xo file
├── app
│   ├── README.MD
│   ├── adas_detection                      # Adas detection application
│   │   ├── model
│   │   ├── src
│   │   ├── adas_detection                  # pre-compiled executable file for zcu102
│   │   └── build.sh
│   ├── resnet50                            # resnet50 application
│   │   ├── model
│   │   ├── src
│   │   ├── resnet50                        # pre-compiled executable file for zcu102
│   │   └── build.sh
│   ├── resnet50_int8                       # resnet50_int8 application
│   │   ├── model
│   │   ├── src
│   │   ├── resnet50_int8                   # pre-compiled executable file for zcu102
│   │   └── build.sh
│   └── resnet50_jpeg                       # resnet50_jpeg application
│       ├── model
│       ├── src
│       ├── libs/libjfif                    # jpeg accelerator lib files
│       ├── resnet50_jpeg                   # pre-compiled executable file for zcu102
│       └── build.sh
└── proj
    ├── build                               # Build TRD flow using source files
    │   ├── classification-pre_DPUv2        # zcu102: sd card image generation for classification example
    │   ├── classification-pre_DPUv3int8    # alveo U200: xclbin generation for classification example    
    │   └── detection-pre_DPUv2             # zcu102: sd card image generation for detection example
    └── pre-built                           # Build using Pre-processor sources & pre-built DPU  
        ├── classification-pre_DPUv2        # zcu102: sd card image generation for classification example
        ├── detection-pre_DPUv2             # zcu102: sd card image generation for detection example
        ├── classification-pre_DPUv3e       # alveo U50: xclbin generation for classification example
        └── detection-pre_DPUv3e            # alveo U50: xclbin generation for detection example

```

## 4 WAA-TRD run

### 4.1 Overview
This tutorial contents information about:
- Build flow for ZCU102 & Alveo-U200 
- Pre-built flow for ZCU102 & Alveo-U50
- Run classification & detection applications

------

### 4.2 Design files
Source  files location for DPU & Pre-processor IP is as below
- DPUv2 IP: `~/Vitis-AI/dsa/DPU-TRD/dpu-ip`
- Pre-processor IP: `~/Vitis-AI/dsa/WAA-TRD/accel/classification-pre`, `classification-pre_int8`, `classification-pre_jpeg` & `detection-pre`

### 4.3 Build and run the application
Build flow for ZCU102 & Alveo-U200

| No. | Build flow                    | Device     | H/W Accelerated Functions                                                        | Documentation                                 |
|-----|-------------------------------|------------|----------------------------------------------------------------------------------|-----------------------------------------------|
| 1   | classification-pre_DPUv2      | ZCU102     | resize, mean subtraction & DPUv2                                                 | [WAA-TRD/proj/build/classification-pre_DPUv2/README](./proj/build/classification-pre_DPUv2/README.md)                      |
| 2   | detection-pre_DPUv2           | ZCU102     | resize, mean subtraction, scale & DPUv2                                          | [WAA-TRD/proj/build/detection-pre_DPUv2/README](./proj/build/detection-pre_DPUv2/README.md)                      |
| 3   | classification-pre_DPUv3int8  | Alveo-U200 | resize, mean subtraction & DPUv3int8                                             | [WAA-TRD/proj/build/classification-pre_DPUv3int8/README](./proj/build/classification-pre_DPUv3int8/README.md)                      |


Pre-built flow for ZCU102 & Alveo-U50

| No. | Pre-built flow                | Device     | H/W Accelerated Functions                                                        | Documentation                                 |
|-----|-------------------------------|------------|----------------------------------------------------------------------------------|-----------------------------------------------|
| 1   | classification-pre_DPUv2      | ZCU102     | JPEG decoder, YUV to RGB conversion, resize, mean subtraction & DPUv2            | [WAA-TRD/proj/pre-built/classification-pre_DPUv2/README](./proj/pre-built/classification-pre_DPUv2/README.md)                      |
| 2   | detection-pre_DPUv2           | ZCU102     | resize, mean subtraction, scale & DPUv2                                          | [WAA-TRD/proj/pre-built/detection-pre_DPUv2/README](./proj/pre-built/detection-pre_DPUv2/README.md)                      |
| 3   | classification-pre_DPUv3e     | Alveo-U50  | resize, mean subtraction & DPUv3e                                                | [WAA-TRD/proj/pre-built/classification-pre_DPUv3e/README](./proj/pre-built/classification-pre_DPUv3e/README.md)                      |
| 4   | detection-pre_DPUv3e          | Alveo-U50  | resize, mean subtraction, scale & DPUv3e                                         | [WAA-TRD/proj/pre-built/detection-pre_DPUv3e/README](./proj/pre-built/detection-pre_DPUv3e/README.md)                     |

## 5 Build with new Pre-processing Accelerator
For Classification & Detection example pre-processor accelerator please refer to [classification-preprocess README](./accel/classification-pre/README.md) & [detection-preprocess README](./accel/detection-pre/README.md) respectively. Array2xfMat, xfMat2hlsStrm & xf::cv::preProcess are manditory submodules in the Pre-process accelerator. As long as the interface of the pp_pipeline_accel function remains same, user can add or remove other submodules depending upon the Deep Neural Network's pre-process requirements.

In this section, example is provided for integrating new Pre-procsssing accelerator with DPU for classification example. 

Provided Resnet50 classification examples uses caffe resnet50 model `WAA-TRD/app/resnet50_waa/resnet50.xmodel`. In this model, pre-processing components are image resize and mean sub operation. Here Bilinear Interpolation is used in the resize. User can changes interpolation type to Nearest Neighbor by modifying line no 61 of accel file `WAA-TRD/accel/classification-pre/xf_pp_pipeline_accel.cpp` as below.
```
61	xf::cv::resize<0,TYPE,HEIGHT,WIDTH,NEWHEIGHT,NEWWIDTH,NPC_T,MAXDOWNSCALE> (imgInput0, out_mat);
```

Note that first template parameter is resize INTERPOLATION type.

// 0 - Nearest Neighbor Interpolation

// 1 - Bilinear Interpolation

// 2 - AREA Interpolation

 After modifying accel file please follow the section 4.3 to integrate new Pre-processing accelerator with DPU and run classification example.  
