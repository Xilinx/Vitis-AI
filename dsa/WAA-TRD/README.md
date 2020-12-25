# WAA-TRD: Whole Application Acceleration - Target Reference Design

## Table of Contents

- [1 Introduction](#1-Introduction)
- [2 Background](#2-Background)
- [3 Directory structure introduction](#3-Directory-structure-introduction)
- [4 WAA-TRD run](#4-WAA-TRD-run)
    - [4.1 Overview](#4.1-Overview)
    - [4.2 Design Files](#42-design-files)
    - [4.3 (Optional) Cross-compile WAA-TRD example](#43-(Optional)-Cross-compile-WAA-TRD-example)
    - [4.4 Build and run the application](#44-build-and-run-the-application)
- [5 Build with new Pre-processing Accelerator](#5-Build-with-new-Pre-processing-Accelerator)    


## 1 Introduction

WAA-TRD demonstrates integration of pre/post processing accelerator with DPU(Deep Neural Network DNN accelerator) for Embedded or cloud platform. The pre/post processing accelerator is implemented using Xilinx® [Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) functions. WAA-TRD provides examples for integrating different pre processing involved in the object classification & detection networks with DPU. 

There are two flows being provided 
1.	Build : Both the pre-processing accelerator and DPU are built from sources. This flow uses DPU-TRD’s make flows.
2.	Pre-built : ML is pre-built and its partial bitfiles are pre-created (using DFx flow). only pre-processing accelerator is built from sources. This flow gives 10x saving in build time.

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
│   │   ├── adas_detection_waa              # pre-compiled executable file
│   │   └── build.sh
│   └── resnet50_waa                        # resnet50 application
│       ├── model
│       ├── src
│       ├── resnet50_waa                    # pre-compiled executable file
│       └── build.sh
└── proj
    ├── build                               # Build TRD flow using source files
    │   ├── classification-pre_DPUv2        # sd card image generation for classification example
    │   │   ├── config_file                 # config file to integrate 2 DPU instances    
    │   │   ├── kernel_xml
    │   │   │   ├── dpu
    │   │   │   └── sfm
    │   │   ├── scripts        
    │   │   ├── scripts_gui            
    │   │   ├── syslink                     # postlink tcl file    
    │   │   ├── Makefile
    │   │   ├── build_classification_pre.sh    
    │   │   └── dpu_conf.vh                 # dpu configuration file
    │   └── detection-pre_DPUv2             # sd card image generation for detection example
    │       ├── config_file                 # config file to integrate 2 DPU instances   
    │       ├── kernel_xml
    │       │   ├── dpu
    │       │   └── sfm
    │       ├── scripts        
    │       ├── scripts_gui            
    │       ├── syslink                     # postlink tcl file    
    │       ├── Makefile
    │       ├── build_detection_pre.sh    
    │       └── dpu_conf.vh                 # dpu configuration file    
    └── pre-built                           # Build using Pre-processor sources & pre-built DPU  
        ├── classification-pre_DPUv2        # sd card image generation for classification example
        │   └── run.sh
        └── detection-pre_DPUv2             # sd card image generation for detection example
            └── run.sh        

```

## 4 WAA-TRD run

### 4.1 Overview
This tutorial contents information about:
- Build & Pre-built flow for ZCU102
- Run classification & detection applications

------

### 4.2 Design files
Source  files location for DPU & Pre-processor IP is as below
- DPU IP: `~/Vitis-AI/dsa/DPU-TRD/dpu-ip`
- Pre-processor IP: `~/Vitis-AI/dsa/WAA-TRD/accel/classification-pre` & `detection-pre`

### 4.3 (Optional) Cross-compile WAA-TRD example
* Download the [sdk-2020.2.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2020.2.0.0.sh)

* Install the cross-compilation system environment, follow the prompts to install. 

    **Please install it on your local host linux system, not in the docker system.**
    ```
    ./sdk-2020.2.0.0.sh
    ```
    Note that the `~/petalinux_sdk` path is recommended for the installation. Regardless of the path you choose for the installation, make sure the path has read-write permissions. 
Here we install it under `~/petalinux_sdk`.

* When the installation is complete, follow the prompts and execute the following command.
    ```
    source ~/petalinux_sdk/environment-setup-aarch64-xilinx-linux
    ```
    Note that if you close the current terminal, you need to re-execute the above instructions in the new terminal interface.

* Download the [vitis_ai_2020.2-r1.3.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2020.2-r1.3.0.tar.gz) and install it to the petalinux system.
    ```
    tar -xzvf vitis_ai_2020.2-r1.3.0.tar.gz -C ~/petalinux_sdk/sysroots/aarch64-xilinx-linux
    ```

* Cross compile `resnet50_waa` example.
    ```
    cd  ~/Vitis-AI/dsa/WAA-TRD/app/resnet50_waa
    bash -x build.sh
    ```
   Cross compile `adas_detection_waa` example.
    ```
    cd  ~/Vitis-AI/dsa/WAA-TRD/app/adas_detection_waa
    bash -x build.sh
    ``` 	
    If the compilation process does not report any error and the executable file `resnet50_waa` & `adas_detection_waa` are generated in the respective example folder, then the host environment is installed correctly.


### 4.4 Build and run the application

### Build flow- Build hardware design from sources and run the application.
- For classification example, please refer to [WAA-TRD/proj/build/classification-pre_DPUv2/README](./proj/build/classification-pre_DPUv2/README.md) file

- For detection example, please refer to [WAA-TRD/proj/build/detection-pre_DPUv2/README](./proj/build/detection-pre_DPUv2/README.md) file

### Pre-built flow- DPU is pre-built and only pre-processing accelerator is built from sources. 


- For classification example, please refer to [WAA-TRD/proj/pre-built/classification-pre_DPUv2/README](./proj/pre-built/classification-pre_DPUv2/README.md) file

- For detection example, please refer to [WAA-TRD/proj/pre-built/detection-pre_DPUv2/README](./proj/pre-built/detection-pre_DPUv2/README.md) file

## 5 Build with new Pre-processing Accelerator
In this section, example is provided for integrating new Pre-procsssing accelerator with DPU.

Provided Resnet50 classification examples uses caffe resnet50 model `WAA-TRD/app/resnet50_waa/resnet50.xmodel`. In this model, pre-processing components are image resize and mean sub operation. Here Bilinear Interpolation is used in the resize. User can changes interpolation type to Nearest Neighbor by modifying line no 61 of accel file `WAA-TRD/accel/classification-pre/xf_pp_pipeline_accel.cpp` as below.

```
61	xf::cv::resize<0,TYPE,HEIGHT,WIDTH,NEWHEIGHT,NEWWIDTH,NPC_T,MAXDOWNSCALE> (imgInput0, out_mat);
```

Note that first template parameter is resize INTERPOLATION type.

// 0 - Nearest Neighbor Interpolation

// 1 - Bilinear Interpolation

// 2 - AREA Interpolation

 After modifying accel file please follow the section 4.4 to integrate new Pre-processing accelerator with DPU and run classification example.  
