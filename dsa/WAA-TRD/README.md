# Zynq UltraScale＋ MPSoC WAA-TRD

## Table of Contents

- [1 Introduction](#1-Introduction)
- [2 Background](#2-Background)
- [3 Directory structure introduction](#3-Directory-structure-introduction)
 
- [4 WAA-TRD run](#4-WAA-TRD-run)
    - [4.1 Overview](#4.1-Overview)
    - [4.2 Software Tools and System Requirements](#42-software-tools-and-system-requirements)
        - [4.2.1 Hardware](#421-hardware)
        - [4.2.2 Software](#422-software)
    - [4.3 Design Files](#43-design-files)
    - [4.4 Tutorial](#44-tutorial)
        - [4.4.1 Board Setup](#441-board-setup)
        - [4.4.2 Build and run the application](#442-build-and-run-the-application)
    - [4.5 Run with new Pre-processing Accelerator](#45-Run-with-new-Pre-processing-Accelerator)    


## 1 Introduction

WAA-TRD demonstrates integration of pre/post processing accelerator with DPU(Deep Neural Network DNN accelerator) for Embedded platform. The pre/post processing accelerator is implemented using Xilinx® [Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) functions. WAA-TRD provides examples for integrating different pre processing involved in the object classification & detection networks with DPU. 

There are two flows being provided 
1.	Build : Both the pre-processing accelerator and DPU are built from sources. This flow uses DPU-TRD’s make flows.
2.	Pre-built : DPU is pre-built and only pre-processing accelerator is built from sources.  This flow gives 10x saving in build time.

## 2 Background
Input images are preprocessed  before being fed for inference of different deep neural networks. The pre-processing steps vary from network to network. For example, for classification networks like Resnet-50 the input image is resized to 224 x 224 size and then channel-wise mean subtraction is performed before feeding the data to the DNN accelerator. For detection networks like YOLO v3 the input image is resized to 256 x 512 size using letterbox before feeding the data to the DNN accelerator. 


[Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) provides functions optimized for FPGA devices that are drop-in replacements for standard OpenCV library functions. This application demonstrates how Vitis Vision library functions can be used to accelerate pre-processing.

## 3 Directory structure introduction
--------------------------------------------------

```
WAA-TRD
├── README.md
├── accel
│   ├── DPUv2_B4096                         # DPU Accelerator xo file used in the Pre-built flow
│   │   └── dpu.xo
│   ├── classification-pre                  # Pre-processing Accelerator for Resnet50 application
│   └── detection-pre                       # Pre-processing Accelerator for Adas detection application
├── app
│   ├── README.MD
│   ├── adas_detection_waa                  # Adas detection application
│   │   ├── model
│   │   ├── src
│   │   ├── adas_detection_waa
│   │   └── build.sh
│   └── resnet50_waa                        # resnet50 application
│       ├── model
│       ├── src
│       ├── resnet50_waa
│       └── build.sh
└── proj
    ├── build                               # sd card generation flow using source files
    │   ├── classification-pre_DPUv2            
    │   │   ├── config_file                 # config file to integrate 2 DPU    
    │   │   ├── kernel_xml
    │   │   │   ├── dpu
    │   │   │   └── sfm
    │   │   ├── scripts        
    │   │   ├── scripts_gui            
    │   │   ├── syslink                     # postlink tcl file    
    │   │   ├── Makefile
    │   │   ├── build_classification_pre.sh    
    │   │   └── dpu_conf.vh                 # dpu configuration file
    │   └── detection-pre_DPUv2
    │       ├── config_file                 # config file to integrate 2 DPU    
    │       ├── kernel_xml
    │       │   ├── dpu
    │       │   └── sfm
    │       ├── scripts        
    │       ├── scripts_gui            
    │       ├── syslink                     # postlink tcl file    
    │       ├── Makefile
    │       ├── build_detection_pre.sh    
    │       └── dpu_conf.vh                 # dpu configuration file    
    └── pre-built                           # sd card generation flow using pre-built DPU IP
        ├── classification-pre_DPUv2
        │   └── run.sh
        └── detection-pre_DPUv2
            └── run.sh        

```

## 4 WAA-TRD run

### 4.1 Overview
This tutorial content information about:
- How to set up the ZCU102 evaluation board
- Build & Pre-built flow
- Run classification & detection application

------

### 4.2 Software Tools and System Requirements

#### 4.2.1 Hardware

Required:

- ZCU102 evaluation board

- Micro-USB cable, connected to laptop or desktop for the terminal emulator

- SD card

#### 4.2.2 Software

  Required:
  - Vitis 2020.2[Vitis Core Development Kit](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/2020-1.html) 
  - [Silicon Labs quad CP210x USB-to-UART bridge driver](http://www.silabs.com/products/mcu/Pages/USBtoUARTBridgeVCPDrivers.aspx)
  - Serial terminal emulator e.g. [teraterm](http://logmett.com/tera-term-the-latest-version)
  - [XRT 2020.2](https://github.com/Xilinx/XRT/tree/2020.2)
  - [zcu102 base platform](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms.html)
  - [mpsoc common system](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zynqmp-common-v2020.1.tar.gz)


###### **Note:** The user can also refer the [zcu102 dpu platform](https://github.com/Xilinx/Vitis_Embedded_Platform_Source/tree/master/Xilinx_Official_Platforms/zcu102_dpu), The github page includes all the details, such as how to generage the zcu102 dpu platform, how to create the SD card after compiling the DPU project.
------


### 4.3 Design files
Source  files location for DPU & Pre-processor IP is as below
- DPU IP: `Vitis-AI/dsa/DPU-TRD/dpu-ip`
- Pre-processor IP: `Vitis-AI/dsa/WAA-TRD/accel/classification-pre` & `detection-pre`

### 4.4 Tutorial

#### 4.4.1 Board Setup

###### Required:

- Connect power supply to 12V power connector.

- Connect micro-USB cable to the USB-UART connector; use the following settings for your terminal emulator:

  - Baud Rate: 115200
  - Data: 8 bit
  - Parity: None
  - Stop: 1 bit
  - Flow Control: None

- Insert SD card.

###### Jumpers & Switches:

  - Set boot mode to SD card:
    - Rev 1.0: SW6[4:1] - **off,off,off, on**
    - Rev D2: SW6[4:1] - **on, off on, off**


#### 4.4.2 Build and run the application

The following tutorials assume that the $TRD_HOME environment variable is set as given below.

```
%export TRD_HOME =<Vitis AI path>/WAA-TRD
```

###### **Note:** It is recommended to follow the build steps in sequence.

We need install the Vitis Core Development Environment.

The following tutorials assume that the Vitis and XRT environment variable is set as given below.

Open a linux terminal. Set the linux as Bash mode.

```
% source <vitis install path>/Vitis/2020.2/settings64.sh
% source opt/xilinx/xrt/setup.sh
% cd $TRD_HOME/
% gunzip <mpsoc common system>/xilinx-zynqmp-common-v2020.2/rootfs.ext4.gz
% export EDGE_COMMON_SW=<mpsoc common system>/xilinx-zynqmp-common-v2020.2 
% export SDX_PLATFORM=<zcu102 base platform path>/xilinx_zcu102_base_202020_1/xilinx_zcu102_base_202020_1.xpfm
% export SYSROOT=<vitis install path>/internal_platforms/sw/zynqmp/xilinx-zynqmp/sysroots/aarch64-xilinx-linux/
```
Note that **mpsoc common system** should be downloaded in the 4.2.2 chapter. 


### Build flow- Build hardware design from sources and run the application.
- For classification example, please refer to [WAA-TRD/proj/build/classification-pre_DPUv2/README](./proj/build/classification-pre_DPUv2/README.md) file

- For detection example, please refer to [WAA-TRD/proj/build/detection-pre_DPUv2/README](./proj/build/detection-pre_DPUv2/README.md) file

### Pre-built flow- DPU is pre-built and only pre-processing accelerator is built from sources. 

Download [bin.tar](https://www.xilinx.com/). Untar the packet and copy `bin` folder to `Vitis-AI/dsa/WAA-TRD/`. 

- For classification example, please refer to [WAA-TRD/proj/pre-built/classification-pre_DPUv2/README](./proj/pre-built/classification-pre_DPUv2/README.md) file

- For detection example, please refer to [WAA-TRD/proj/pre-built/detection-pre_DPUv2/README](./proj/pre-built/detection-pre_DPUv2/README.md) file

### 4.5 Build with new Pre-processing Accelerator
In this section, example is provided for integrating new Pre-processing accelerator with DPU.

Provided Resnet50 classification examples uses caffe resnet50 model `WAA-TRD/app/resnet50_waa/resnet50.xmodel`. In this model, pre-processing components are image resize and mean sub operation. Here Bilinear Interpolation is used in the resize. User can changes interpolation type to Nearest Neighbor by modifying line no 61 of accel file `WAA-TRD/accel/classification-pre/xf_pp_pipeline_accel.cpp` as below. After modifying accel file please follow the section 4.4 to integrate new Pre-processing accelerator with DPU and run classification example.  

```
61	xf::cv::resize<0,TYPE,HEIGHT,WIDTH,NEWHEIGHT,NEWWIDTH,NPC_T,MAXDOWNSCALE> (imgInput0, out_mat);
```

Note that first template parameter is resize INTERPOLATION type.

// 0 - Nearest Neighbor Interpolation

// 1 - Bilinear Interpolation

// 2 - AREA Interpolation
