# Zynq UltraScale＋ MPSoC DPU TRD Vitis 2019.2

## Table of Contents

- [1 Revision History](#1-revision-history)
- [2 Overview](#2-overview)
- [3 Software Tools and System Requirements](#3-software-tools-and-system-requirements)
    - [3.1 Hardware](#31-hardware)
    - [3.2 Software](#32-software)
- [4 Design Files](#4-design-files)
    - [Design Components](#design-components)
- [5 Tutorials](#5-tutorials)
	- [5.1 Board Setup](#51-board-setup)
	- [5.2 Build and Run TRD Flow](#52-build-and-run-trd-flow)
		- [5.2.1 Build the Hardware Design](#521-build-the-hardware-design)
   		- [5.2.2 Get HWH File](#522-get-hwh-file)
   		- [5.2.3 Run Resnet50 Example](#523-resnet50-example)
	- [5.3 Configurate the DPU](#3-configurate-the-dpu)
		- [5.3.1 Set the DPU Core Number](#531-set-dpu-core-number)
		- [5.3.2 Modify the Parameters](#532-modify-the-parameters)
		- [5.3.3 Specify Connectivity for DPU Ports](#533-specify-connectivity-for-dpu-ports)
   	- [5.4 Integrate the DPU in customer platform](#54-integrate-the-dpu-in-customer-platform)
	- [5.5 Integrate the DPU for zcu102 and zcu104 AI-SDK release](#55-integrate-the-dpu-for-zcu102-and-zcu104-ai-sdk-released)
		- [5.5.1 Configue the zcu102 released project ](#551-configue-the-zcu102-released-project)
		- [5.5.2 Configue the zcu104 released project ](#552-configue-the-zcu104-released-project)

## 1 Revision History

This wiki page complements the Vitis 2019.2 version of the DPU TRD.

Change Log:

-  The first version of Vitis DPU TRD

------

## 2 Overview



This tutorial contains information about:

- How to set up the ZCU102 evaluation board and run the TRD.
- How to change the Configuration of DPU.
- How to integrate the DPU in the customer platform in vitis 2019.2 environment.

------

## 3 Software Tools and System Requirements

### 3.1 Hardware

Required:

- ZCU102 evaluation board

- Micro-USB cable, connected to laptop or desktop for the terminal emulator

- SD card

### 3.2 Software

  Required:
  - Vitis 2019.2[Vitis Core Development Kit](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html) 
  - [Silicon Labs quad CP210x USB-to-UART bridge driver](http://www.silabs.com/products/mcu/Pages/USBtoUARTBridgeVCPDrivers.aspx)
  - Serial terminal emulator e.g. [teraterm](http://logmett.com/tera-term-the-latest-version)
  - [XRT 2019.2](https://github.com/Xilinx/XRT/tree/2019.2)
  - [zcu102 dpu platform](https://www.xilinx.com/bin/public/openDownload?filename=zcu102_dpu_2019.2.zip)
  - [Vitis AI Library](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Library) to configure DPU in Vitis AI Library ZCU102 and ZCU104 pacakge, Optional


###### **Note:** The user can also refer the [zcu102 dpu platform](https://github.com/Xilinx/Vitis_Embedded_Platform_Source/tree/master/Xilinx_Official_Platforms/zcu102_dpu), The github page includes all the details, such as how to generage the zcu102 dpu platform, how to create the SD card after compiling the DPU project.
------

## 4 Design Files

### Design Components

The top-level directory structure shows the the major design components. The TRD directory is provided with a basic README and legal notice file.

###### **Note:** The xdpu/dpu_ip and xdpu/prj/Vitis(inclduing the kernel_xml,dpu_config.vh,scripts) are needed, if the user add the DPU and softmax in own project.

```
DPU_TRD       
├── dpu_ip                              # rtl kernel
│   ├── dpu_eu_v3_2_0
│   │   ├── bd
│   │   ├── component.xml
│   │   ├── doc
│   │   ├── gui
│   │   ├── hdl
│   │   ├── inc
│   │   ├── ttcl
│   │   ├── xci
│   │   └── xgui
│   └── Vitis
│       ├── dpu
│       └── sfm
├── app       
│   ├── models
│   ├── img 
│   └── resnet50.tar.gz             # resnet50 application
└── prj 
    └── Vitis
        │        
        ├── kernel_xml                                        
        │   ├── dpu
        │   └── sfm 
        ├── Makefile
        ├── dpu_conf.vh                 # dpu configuration file
        ├── config_file                 # config file
        │   ├── prj_config              # integrate 2DPU 
        │   ├── prj_config_102_3dpu     # integrate 3DPU on zcu102
        │   ├── prj_config_104_2dpu     # integrate 2DPU on zcu104
        │   └── prj_config_1dpu         # integrate 1DPU on zcu104
        ├── scripts
        └── README.md

```

## 5 Tutorials

### 5.1 Board Setup

###### Required:

- Connect power supply to 12V power connector.

- Connect micro-USB cable to the USB-UART connector; use the following settings for your terminal emulator:

  - Baud Rate: 115200
  - Data: 8 bit
  - Parity: None
  - Stop: 1 bit
  - Flow Control: None

- Insert SD card (FAT formatted) with binaries copied from $TRD_HOME/images directory.

###### Jumpers & Switches:

  - Set boot mode to SD card:
    - Rev 1.0: SW6[4:1] - **off,off,off, on**
    - Rev D2: SW6[4:1] - **on, off on, off**



To run the pre-built SD card image , follow the instructions on [5.2.3](#523-run-flow-tutorial) in this page

### 5.2 Build and Run Flow

The following tutorials assume that the $TRD_HOME environment variable is set as given below.

```
%export TRD_HOME =<Vitis AI path>/DPU-TRD
```

###### **Note:** It is recommended to follow the build steps in sequence.

#### 5.2.1 Building the Hardware Design

We need install the Vitis Core Development Environment.

We prepare the zcu102_dpu platform in the vitis TRD project. The platform include all the libs that needed.

The following tutorials assume that the Vitis and XRT environment variable is set as given below.

Open a linux terminal. Set the linux as Bash mode.

```
% source <vitis install path>/Vitis/2019.2/settings64.sh

% source opt/xilinx/xrt/setup.sh
```

The default setting of DPU is **B4096** with RAM_USAGE_LOW, CHANNEL_AUGMENTATION_ENABLE, DWCV_ENABLE, POOL_AVG_ENABLE, RELU_LEAKYRELU_RELU6, Softmax. Modify the $TRD_HOME/prj/Vitis/dpu_conf.vh file can change the default settings.

Build the hardware design.

```
% cd $TRD_HOME/prj/Vitis

% export SDX_PLATFORM=<vitis install path>/platform/zcu102_dpu/zcu102_dpu.xpfm

% make KERNEL=DPU_SM DEVICE=zcu102
```

Generated SD card files are in **$TRD_HOME/prj/Vitis/binary_container_1/sd_card**.

#### 5.2.2 Get HWH File 

HWH file is a important file that need by the vai-c tool. The file has been created when compile by the Vitils tool. It works together with vai-c to support model compilation under various DPU configurations.

The HWH file could be simply get from $TRD_HOME/prj/Vitis/binary_container_1/sd_card

The user can also get the HWH file in the following path.

$TRD_HOME/prj/Vitis/binary_container_1/link/vivado/vpl/prj/prj.srcs/sources_1/bd/zcu102_dpu/hw_handoff/zcu102_dpu.hwh

Different Platform will get different name of HWH file.
 
#### 5.2.3 Run Resnet50 Example 

The TRD project has generated the matching model file in $TRD_HOME/prj/app/ path as the default settings. If the user change the DPU settings. The model need to be created again.

This part is about how to run the Resnet50 example from the source code.

The user must create the SD card. Refer section "Configuring SD Card ext File System Boot" in page 65 of [ug1144](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2019_2/ug1144-petalinux-tools-reference-guide.pdf)for Petalinux 2019.2:

Copy the whole files in **$TRD_HOME/prj/Vitis/binary_container_1/sd_card** execpt the rootfs.tar.gz fiel to BOOT partition.

Extract the rootfs.tar.gz in the RootFs folder 

Copy the directory **$TRD_HOME/app** to the BOOT partition of the SD Card.

After the linux boot, run:

```

% cp /mnt/dpu.xclbin /usr/lib

% tar -xvf /mnt/app/resnet50.tar.gz -C ~

% cp /mnt/app/models/resnet50.elf ~

% cp -r /mnt/app/img ~

% cd ~

% env LD_LIBRARY_PATH=samples/lib samples/bin/resnet50 img/bellpeppe-994958.JPEG

Expect: 
score[945]  =  0.992235     text: bell pepper,
score[941]  =  0.00315807   text: acorn squash,
score[943]  =  0.00191546   text: cucumber, cuke,
score[939]  =  0.000904801  text: zucchini, courgette,
score[949]  =  0.00054879   text: strawberry,

```


###### **Note:** The $TRD_HOME/prj/app resenet50 test case can support both Vitis and Vivado flow . If you want to run other network. Please refer to the [Vitis AI Github](https://github.com/Xilinx/Vitis-AI) and [Vitis AI User Guide](http://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf).


### 5.3 Change the Configuration


The DPU IP provides some user-configurable parameters to optimize resource utilization and customize different features. Different configurations can be selected for DSP slices, LUT, block RAM(BRAM), and UltraRAM utilization based on the amount of available programmable logic resources. There are also options for addition functions, such as channel augmentation, average pooling, depthwise convolution.

The TRD also support the softmax function.
   
For more details about the DPU, please read [DPU IP Product Guide](https://www.xilinx.com/cgi-bin/docs/ipdoc?c=dpu;v=latest;d=pg338-dpu.pdf)

 
#### 5.3.1 Set the DPU Core Number

The DPU core number is set 2 as default setting. Modify the prj_config file in the [connectivity] part.

```
nk=dpu_xrt_top:2
```
The project will integrate 2 DPU. The user can delete this property, Then the project will integrate 1 DPU. Change the number 2 to others, The project will integrate DPU number as you want.

###### **Note:** The DPU needs lots of LUTs and RAMs. Use 3 or more DPU numbers may cause the resourse and timing issue.


#### 5.3.2 Modify the Parameters

The default setting is B4096 for ZCU102. Read the dpu_conf.vh file to get the details of DPU. You can get all the configurations form [PG338](https://www.xilinx.com/support/documentation/ip_documentation/dpu/v3_2/pg338-dpu.pdf) 

Modify the $TRD_HOME/prj/Vitis/dpu_conf.vh file to modify the configuration. 

The TRD supports to modify the following parameters.

- Architecture
- URAM Number
- RAM Usage 
- Channel Augmentation 
- DepthwiseConv
- AveragePool
- ReLU Type
- DSP Usage
- Device 
- Softmax

#### Architecture

The DPU can configurate hardware architecture, including: **B512, B800, B1024, B1152, B1600, B2304, B3136, B4096**.

If you want to choose the B4096 DPU. Need to set like this.

```
`define B4096
```
#### URAM Number

Enable the URAM, The DPU will replace the bram to the uram.

```
`define URAM_ENABLE
```
Disable the URAM
```
`define URAM_DISABLE
```

For example, Use the DPU B4096 in zcu104 board. Setting as following.

```
`define B4096
`define URAM_ENABLE
```

There are some recommended uram numbers for different sizes.

| |B512|B800|B1024|B1152|B1600|B2304|B3136|B4096|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|U_BANK_IMG|2|2|4|2|4|4|4|5|
|U_BANK_WGT|9|11|9|13|11|13|15|17|
|U_BANK_BIAS|1|1|1|1|1|1|1|1|

Modify the **$TRD_HOME/prj/Vitis/dpu_conf.vh** file to enable or disable the URAM function and change the URAM numbers.

Change the URAM numbers.

```
`ifdef URAM_ENABLE
    `define def_UBANK_IMG_N          5
    `define def_UBANK_WGT_N          17
    `define def_UBANK_BIAS           1
`elsif URAM_DISABLE
    `define def_UBANK_IMG_N          0
    `define def_UBANK_WGT_N          0
    `define def_UBANK_BIAS           0
`endif
```
#### RAM Usage

There are two options of RAM usage. High RAM Usage means that the on-chip memory block will be larger, allowing the DPU more flexibility to handle the internediate data. If the RAM is limited, Please the low RAM Usage.

#### Channel Augmentation

Enable 
```
`define CHANNEL_AUGMENTATION_ENABLE
```
Disable 
```
`define CHANNEL_AUGMENTATION_DISABLE
```

#### DepthwiseConv

Enable
```
`define DWCV_ENABLE
```
Disable
```
`define DWCV_DISABLE
```

#### AveragePool

Enable
```
`define POOL_AVG_ENABLE
```
Disable
```
`define POOL_AVG_DISABLE
```

#### RELU Type

There are two  options of RELU Type, including: RELU_RELU6,RELU_LEAKRELU_RELU6. We recommend use the RELU_LEAKRELU_RELU6.
```
`define RELU_LEAKYRELU_RELU6
```

#### DSP Usage
Setting High will cost more DSP rescources than LOW.
```
`define DSP48_USAGE_HIGH
```
LOW
```
`define DSP48_USAGE_LOW
```

#### Softmax

The TRD support the softmax function. The TRD has included the softmax rtl kernel.

Only use the DPU
```
make KERNEL=DPU
```
Use the DPU and Softmax
```
make KERNEL=DPU_SM
```

####5.3.3 Specify Connectivity for DPU Ports

Need to specify connectivity to the various ports in the system for the DPU. Open the file **prj_config** in a text editor. Refer the vitis document. Using the following comment to check the ports of platform.

```
% platforminfo -p <platform path>/zcu104_revmin/zcu104_revmin.xpfm
```

The information of platform is shown in the below figure.

![information platform](./doc/5.3.3.png)

The default port connection is shown below.

|IP|Port|Connection|
|:---|:---|:---|
| |M_AXI_GP0|HP0|
|dpu_xrt_top_1|M_AXI_HP0|HP1|
| |M_AXI_HP2|HP2|
| |M_AXI_GP0|HP0|
|dpu_xrt_top_2|M_AXI_HP0|HPC0|
| |M_AXI_HP2|HPC1|


If the platform doesn't have enough port to connect the port of DPU. The ports can share with other IPs.


------

### 5.4 Integrate the DPU in customer platform

Refer the UG1146 or [Vitis_Embedded_Platform_Source](https://github.com/Xilinx/Vitis_Embedded_Platform_Source) to create the vitis platform. Modify the **SDX_PLATFORM** to specify the user platform.

```
% export SDX_PLATFORM=<user platform path>/user_platform.xpfm
```

If user just want to integrate the DPU IP in user platform. The related files need to be copied incuding the following path.

$TRD_HOME/prj/Vitis
$TRD_HOME/dpu_ip

The $TRD/prj/Vitis/strip_interconects.tcl file is for reducing the resources and not necessary. It can be detelted. The relevant statement need to be deleted inthe makefile at the same time.

Delete the following property.

```
--xp param:compiler.userPostSysLinkTcl=${DIR_PRJ}/strip_interconnects.tcl 
```

If the user changes the directroy structure, the relative paths in the script files need to be modified.

The other steps refer the 5.2 chapter. 

If you meet some timing issues. you can modify the [vivado] part of prj_config file (**prop=run.impl_1.strategy=Performance_Explore**) to change another implementation strategy and re-compile the project.


------

### 5.5 Integrate the DPU for Vitis AI Library release

This chapter introduces how to configue the project for [Vitis AI Library](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Library) released package for ZCU102 and ZCU104.

#### 5.5.1 Configue the zcu102 released project

steps:

1.Modify the Makefile file
```
line13: --config ${TRD_HOME}/prj/Vitis/config_file/prj_config_102_3dpu
```
2.
```
% make KERNEL=DPU_SM DEVICE=zcu102
```


#### 5.5.2 Configue the zcu104 released project

steps:

1.Modify the Makefile file
```
line13: --config ${TRD_HOME}/prj/Vitis/config_file/prj_config_104_2dpu
```
2.Enable the URAM and modify the RAM USAGE

Need to modify the dpu_conf.vh file
```
line52:`define URAM_ENABLE
line73:`define RAM_USAGE_HIGH
```
3.
```
% make KERNEL=DPU DEVICE=zcu104
```


