# VCK190 DPUCVDX8G TRD for Vitis AI

**Note1:** In this TRD, platform is based on VCK190 ES1 board.

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
		- [5.2.2 Get Json File](#522-get-json-file)
		- [5.2.3 Run Resnet50 Example](#523-run-resnet50-example)
	- [5.3 Change the Configuration](#53-change-the-configuration)
- [6 Instruction for Changing Platform](#6-instruction-for-changing-platform)
	- [6.1 Interfaces of DPUCVDX8G](#61-interfaces-of-dpucvdx8g)
	- [6.2 Changing Platform](#62-changing-platform)
	- [6.3 Improve Performance](#63-improve-performance)
- [7 Basic Requirement of Platform](#7-basic-requirement-of-platform)
- [8 Vivado Project of TRD Platform1](#8-vivado-project-of-trd-platform1)
- [9 Known Issue ](#9-known-issue)

## 1 Revision History

Change Log:

-  This is the first version of DPUCVDX8G TRD for Vitis AI.
   
------

## 2 Overview

The Xilinx Versal Deep Learning Processing Unit (DPUCVDX8G) is a computation engine optimized for convolutional neural networks.
It includes a set of highly optimized instructions, and supports most convolutional neural networks, such as VGG, ResNet, GoogLeNet, YOLO, SSD, MobileNet, and others.

This tutorial contains information about:

- How to set up the VCK190 evaluation board.
- How to build and run the DPUCVDX8G TRD with VCK190 platform in Vitis 2020.2 environment.
- How to change platform.
------

## 3 Software Tools and System Requirements

### 3.1 Hardware

Required:

- VCK190 evaluation board

- Micro-USB cable, connected to laptop or desktop for the terminal emulator

- SD card 

### 3.2 Software

  Required:
  - Vitis 2020.2[Vitis Core Development Kit](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/2020-2.html) 
  - [XRT 2020.2](https://github.com/Xilinx/XRT/tree/2020.2)
  - Python (version 2.7.5 or 3.6.8)

------

## 4 Design Files

### Design Components

The top-level directory structure shows the the major design components.

```
├── app
├── README.md
├── vck190_platform                        # VCK190 platform folder
│   ├── hw
│   ├── Makefile
│   ├── platform
│   ├── README.md
│   └── sw
├── vitis_prj                              # Vitis project folder
│   ├── Makefile
│   ├── scripts
│   ├── xvdpu
│   └── xvdpu_config.mk
└── xvdpu_ip                               # DPUCVDX8G IP folder
    ├── aie
    └── rtl

```
------

## 5 Tutorials

### 5.1 Board Setup

###### Board jumper and switch settings:

Please make sure the board is set as booting from SD card:



- Remove J326 (7-8) jumper.


- SW11[4:1]- [OFF,OFF,OFF,ON].


- SW1[4:1]- [OFF,OFF,OFF,ON].

### 5.2 Build and Run TRD Flow

The following tutorials assume that the $TRD_HOME environment variable is set as given below.

```
% export TRD_HOME =<Vitis AI path>/XVDPU-TRD
```
We need install the Vitis Development Environment.

And make sure to enable the Early Access devices (Refer to https://www.xilinx.com/member/vck190_headstart.html#started ). 

By adding the following line to each of the tcl scripts:  
```   
 enable_beta_device *
```  
- $HOME/.Xilinx/Vivado/Vivado_init.tcl   
- $HOME/.Xilinx/HLS_init.tcl

**Step1:** Build VCK190 ES1 platform 

Firstly, please build the VCK190 ES1 platform in the folder '$TRD_HOME/vck190_platform', build steps please follow '$TRD_HOME/vck190_platform/README.md".


**Step2:** Setup environment for building DPUCVDX8G

When platform is ready, please set the Vitis and XRT environment variable as given below.

Open a linux terminal. Set the linux as Bash mode.

```
% source <vitis install path>/Vitis/2020.2/settings64.sh

% source opt/xilinx/xrt/setup.sh
```

###### **Note:** It is recommended to follow the build steps in sequence.

#### 5.2.1 Build the Hardware Design

The default setting of DPUCVDX8G is 3 Batch (CPB_N=32), Frequency is 333 MHz, UBANK_IMG_N (=16) and UBANK_WGT_N (=17) are set as the Max value.
Modify file '$TRD_HOME/vitis_prj/xvdpu_config.mk' can change the default settings. 
 

Build the hardware design.

```
% cd $TRD_HOME/vitis_prj

% make files

% make all

```

Generated SD card image:  $TRD_HOME/vitis_prj/package_out/sd_card.img

Implemented Vivado project: $TRD_HOME/vitis_prj/hw/binary_container_1/link/vivado/vpl/prj

**Note1:** With 'make help' to check the detailed information about the commands. 

**Note2:** Implementation strategy is set in the file '$TRD_HOME/vitis_prj/scripts/system.cfg', the default strategy is " prop=run.impl_1.strategy=Performance_ExploreWithRemap ".
           Changing implementation strategy can be done by changing this line in 'system.cfg'. 

		   
#### 5.2.2 Get Json File

The 'arch.json' file is an important file required by Vitis AI. It works together with Vitis AI compiler to support model compilation with various DPUCVDX8G configurations. The 'arch.json' will be generated by Vitis during the compilation of DPUCVDX8G TRD, it can be found in '$TRD_HOME/vitis_prj/package_out/sd_card' .

It can also be found in the following path:

$TRD_HOME/vitis_prj/hw/binary_container_1/link/vivado/vpl/prj/prj.gen/sources_1/bd/vck190*/ip/*_DPUCVDX8G_0/arch.json


#### 5.2.3 Run Resnet50 Example

The TRD project has generated the matching model file in '$TRD_HOME/prj/app/' path for the default settings. If the user change the DPUCVDX8G settings. The model need to be created again.

This part is about how to run the Resnet50 example.

Use the balenaEtcher tool to flash '$TRD_HOME/vitis_prj/package_out/sd_card.img' into SD card, insert the SD card with the image into the destination board and power-on it.
After Linux booting on board, copy the folder '$TRD_HOME/app' in this TRD to the target folder "~/", then run below commands:

```
% cd ~/app/model/

% test_dpu_runner_mt resnet50.xmodel x_0 1

```
Expect result like below: 

```
I0130 22:02:10.517663   755 test_dpu_runner_mt.cpp:399] create runner ... 0/1
I0130 22:02:11.101772   755 performance_test.hpp:73] 0% ...
I0130 22:02:17.101913   755 performance_test.hpp:76] 10% ...
...
I0130 22:03:11.103199   755 performance_test.hpp:76] 100% ...
I0130 22:03:11.103253   755 performance_test.hpp:79] stop and waiting for all threads terminated....
I0130 22:03:11.104063   755 performance_test.hpp:85] thread-0 processes 73776 frames
I0130 22:03:11.104090   755 performance_test.hpp:93] it takes 815 us for shutdown
I0130 22:03:11.104107   755 performance_test.hpp:94] FPS= 1229.55 number_of_frames= 73776 time= 60.0023 seconds.
I0130 22:03:11.104150   755 performance_test.hpp:96] BYEBYE
```
###### **Note:** For running other network. Please refer to the [Vitis AI Github](https://github.com/Xilinx/Vitis-AI) and [Vitis AI User Guide](http://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf).


### 5.3 Change the Configuration

The DPUCVDX8G IP provides some user-configurable parameters, please refer to the document PG389 'Xilinx Versal DPU (DPUCVDX8G) Product Guide' .
In this TRD, user-configurable parameters are in the file '$TRD_HOME/vitis_prj/xvdpu_config.mk'. They are:
- BATCH_N     -- number of batch engine integrated in DPUCVDX8G IP, range from 1 to 6.
- UBANK_IMG_N -- number of IMG BANKs are composed of UltraRAM, the Max is 16.
- UBANK_WGT_N -- number of WGT BANKs are composed of UltraRAM, the Max is 17.
- PL_FREQ     -- Frequency of 'm_axi_aclk', support range from 200M to 333M Hz. 

Other parameters are set by default as below:
- CPB_N               = 32
- LOAD_PARALLEL_IMG   = 2
- SAVE_PARALLEL_IMG   = 2

After changing '$TRD_HOME/vitis_prj/xvdpu_config.mk', type 'make files' and 'make all' to build the design. 

------

## 6 Instruction for Changing Platform

### 6.1 Interfaces of DPUCVDX8G

Interfaces of DPUCVDX8G are listed as below.

- m*_wgt/img/bias/instr_axi -- Master AXI interfaces, connected with LPDDR's NOC (NOC_0 in this TRD platform1)
- s*_ofm_axis     -- Slave AXI-stream interface, connected with AIE (ai_engine_0).
- m*_ifm/wgt_axis -- Master AXI-stream interface, connected with AIE (ai_engine_0).
- m_axi_clk   -- Input clock used for DPUCVDX8G general logic, AXI and AXI-stream interface. Frequency is 333M Hz in this TRD.
- m_axi_aresetn -- Active-Low reset for DPUCVDX8G general logic. 
- s_axi_control  -- AXI lite interface for controlling DPUCVDX8G registors, connected with CIPs through AXI_Smartconnect_IP.
- s_axi_aclk   -- Input clock for S_AXI_CONTROL. Frequency is 150M Hz in this TRD.
- s_axi_aresetn -- Active-Low reset for S_AXI_CONTROL.
- interrupt -- EDGE_RISING interrupt signal generated by DPUCVDX8G.


The connection between AXI-stream interface of DPUCVDX8G and AIE are defined in the '$TRD_HOME/vitis_prj/scripts/DPUCVDX8G_aie.cfg'.

Since the platform in this TRD does not support 'sptag', the connection between Master AXI interface of DPUCVDX8G and LPDDR's NOC are defined in the '$TRD_HOME/vitis_prj/scripts/postlink.tcl'.

For the clock design, please make sure that:
- m_axi_clk of DPUCVDX8G, aclk0 of AIE (ai_engine_0), and clock for the slave AXI interfaces of LPDDR's NOC connected with DPUCVDX8G (aclk12 of NOC_0 in the TRD), should be drived by the same clock source. (333M Hz in this TRD)
- s_axi_aclk for 's_axi_control' should use clock with lower frequency, such as 150M Hz, to get better timing.
- 'AI Engine Core Frequency' should be 4 times of DPUCVDX8G's ap_clk. In this TRD, it is 1333M Hz (4 x 333M). The value of 'AI Engine Core Frequency' can be set in the platform design files or '/vitis_prj/scripts/postlink.tcl'.

### 6.2 Changing Platform

Changing platform needs to modify the path of platform files in the '$TRD_HOME/vitis_prj/Makefile', and disable 'postlink.tcl' (it is specified for the VCK190 platform in this TRD)


- Change the path of 'xpfm' file for varibale 'PLATFORM'
```
  PLATFORM           = */*.xpfm
```
- Change the path of 'rootfs.exts' and 'Image' in the package section (bottom of Makefile) 
```
  --package.rootfs     */rootfs.ext4 \
  --package.sd_file    */Image \
```
- Disable 'postlink.tcl' 
```
#VXXFLAGS                += --xp param:compiler.userPostSysLinkOverlayTcl=$(ABS_PATH)/scripts/post_linker.tcl
```

#### 6.3 Improve Performance

There are DDR and LPDDR on the VCK190 board. For getting better performance, please connect DPUCVDX8G with the LPDDR's NOC. This can be done by platform design (or sptag), or commands in the '/vitis_prj/scripts/postlink.tcl'. 
Also, please do NOC performance tunning to get more better performance. You can refer to Chapter 5 of [PG313 - Versal ACAP Programmable Network on Chip and Integrated Memory Controller v1.0 Product Guide](https://www.xilinx.com/support/documentation/ip_documentation/axi_noc/v1_0/pg313-network-on-chip.pdf)

For 'postlink.tcl', a general way is that firstly disabling the 'postlink.tcl' in the Makefile, let Vitis creating the default block design for your project.
```
  #VXXFLAGS  += --xp param:compiler.userPostSysLinkOverlayTcl=$(ABS_PATH)/scripts/post_linker.tcl
```
Open the block design in the Vivado GUI, do changes as you wanted, and copy the commands in the 'Tcl Console' into your 'postlink.tcl'.
Then enable the 'postlink.tcl' in the makefile, 'make clean' and build the project again. 
```
  VXXFLAGS  += --xp param:compiler.userPostSysLinkOverlayTcl=$(ABS_PATH)/scripts/post_linker.tcl
```
In the 'postlink.tcl' of this TRD, commands are:  
 -  Set 'AI Engine Core Frequency'
 -  Connect master AXI interfaces of DPUCVDX8G directly with LPDDR's NOC (NOC_0)
 -  Set the QoS of NOC_0's slave AXI interfaces 
 -  Set the clock connection
 -  Set the reset connection. 
 ------
 
## 7 Basic Requirement of Platform
For platform which will integrate DPUCVDX8G, the basic requirements are listed as below:
- One 'CIPS' IP.
- One 'NOC' IP with its slave AXI interfaces connected with DPUCVDX8G, and its DDR interface should provide best DDR bandwidth for DPUCVDX8G. For VCK190 board as example, the NOC should have 2x Memory Controller(for 2 LPDDR on board) and 4x MC ports. 
- One 'AI Engine' IP name 'ai_engine_0', and its Core Frequency should be 1333 MHz.
- One 'Clocking Wizard' IP, with at least 2 output clocks for DPUCVDX8G (150 MHz and 333 MHz).   
- Two 'Processor System Resets' IP, for 150 MHz and 333 MHz.
- One 'AXI smartconnect' IP with its master port enabled in the platform (for connection with DPUCVDX8G's 's_axi_control' port), and its slave interface connected to the CIPS master port. 
- One 'AXI interrupt controller' IP with its interrupt port connected to pl_ps_irqXX of CIPS and its slave AXI port connected to the CIPs master with its address mapped to 0xA5000000.    

For the detailed platform design, please refer to VCK190 TRD platform1.

 ------
 
## 8 Vivado Project of TRD Platform1
Source Tcl files for XSA of TRD platform1 are in the folder '/vck190_platform/hw'.

## 9 Known Issue 
Unsupported Models:
- SA_gate_pt
- fadnet

For the configuration 'BATCH_N = 6', due to limited DSP resource in XCVC1902, 'ELEW_MULT_ENA' is disabled, model 'efficientnet-b0_tf2' is not supported.