# VCK190 DPUCVDX8G TRD for Vitis AI

**Note1:** In this TRD, platform is based on VCK190 Prod board.

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
- [7 Basic Requirement of Platform](#7-basic-requirement-of-platform)
- [8 Vivado Project of TRD Platform1](#8-vivado-project-of-trd-platform1)
- [9 Known Issue ](#9-known-issue)

## 1 Revision History

Vitis2.0 Change log:
- Change platform as VCK190-prod 2021.2 version
- Change AI Engine Core Frequency from 1333 MHz to 1250 MHz
- Add XVDPU configuration： C64B1，C64B2，C64B4，C64B5. All supported configurations are C32B[1:6] and C64B[1:5]

Vitis1.4.1 Change log:
- Change platform as VCK190-prod 2021.1 version
- Update scripts to make changing platform more easier, update 'vitis_prj/Makefile' to simple the HW build steps
- Add XVDPU configuration C64B3
------

## 2 Overview

The Xilinx Versal Deep Learning Processing Unit (DPUCVDX8G) is a computation engine optimized for convolutional neural networks.
It includes a set of highly optimized instructions, and supports most convolutional neural networks, such as VGG, ResNet, GoogLeNet, YOLO, SSD, MobileNet, and others.

This tutorial contains information about:

- How to set up the VCK190 evaluation board.
- How to build and run the DPUCVDX8G TRD with VCK190 platform in Vitis environment.
- How to change platform.
------

## 3 Software Tools and System Requirements

### 3.1 Hardware

Required:

- VCK190 Prod evaluation board

- Micro-USB cable, connected to laptop or desktop for the terminal emulator

- SD card 

### 3.2 Software

  Required:
  - Vitis 2021.2
  - XRT 2021.2
  - Python (version 2.7.5 or 3.6.8)
  - csh

------

## 4 Design Files

### Design Components

The top-level directory structure shows the the major design components.

```
├── app
├── README.md
├── vck190_platform                        # VCK190 platform folder
│   ├── LICENSE
│   ├── Makefile
│   ├── overlays
│   ├── petalinux
│   ├── platforms
│   └── README.md
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
% export TRD_HOME =<Vitis AI path>/dsa/XVDPU-TRD
```

**Step1:** Build VCK190 platform 

Firstly, please build the VCK190 platform in the folder '$TRD_HOME/vck190_platform', build steps please follow '$TRD_HOME/vck190_platform/README.md".


**Step2:** Setup environment for building DPUCVDX8G

When platform is ready, please set the Vitis and XRT environment variable as given below.

Open a linux terminal. Set the linux as Bash mode.

```
% source <vitis install path>/Vitis/2021.2/settings64.sh

% source opt/xilinx/xrt/setup.sh
```

###### **Note:** It is recommended to follow the build steps in sequence.

#### 5.2.1 Build the Hardware Design

The default setting of DPUCVDX8G is 3 Batch (CPB_N=32), Frequency is 333 MHz, UBANK_IMG_N (=16) and UBANK_WGT_N (=17) are set as the Max value.
Modify file '$TRD_HOME/vitis_prj/xvdpu_config.mk' can change the settings.  

Build the hardware design.

```
% cd $TRD_HOME/vitis_prj

% make all

```

Generated SD card image:  $TRD_HOME/vitis_prj/package_out/sd_card.img.gz

Implemented Vivado project: $TRD_HOME/vitis_prj/hw/binary_container_1/link/vivado/vpl/prj/prj.xpr

**Note1:** With 'make help' to check the detailed information about the commands. 

**Note2:** Implementation strategy is set in the file '$TRD_HOME/vitis_prj/scripts/system.cfg', the default strategy is " prop=run.impl_1.strategy=Performance_ExploreWithRemap ".
           Changing implementation strategy can be done by changing this line in 'system.cfg'. 

**Note3:** With same configuration of XVDPU, 'libadf.a' file of AIE can be re-used. By disabling the last line of '$TRD_HOME/vitis_prj/Makefile', to save the compile time for re-building the hardware design.
```
# -@rm -rf aie
```
		   
#### 5.2.2 Get Json File

The 'arch.json' file is an important file required by Vitis AI. It works together with Vitis AI compiler to support model compilation with various DPUCVDX8G configurations. The 'arch.json' will be generated by Vitis during the compilation of DPUCVDX8G TRD, it can be found in '$TRD_HOME/vitis_prj/package_out/sd_card' .

It can also be found in the following path:

$TRD_HOME/vitis_prj/hw/binary_container_1/link/vivado/vpl/prj/prj.gen/sources_1/bd/vck190*/ip/*_DPUCVDX8G_1_0/arch.json


#### 5.2.3 Run Resnet50 Example

The TRD project has generated the matching model file in '$TRD_HOME/app' path for the default settings. If changing the settings of DPUCVDX8G, the model need to be created again.

This part is about how to run the Resnet50 example.

Use the balenaEtcher tool to flash '$TRD_HOME/vitis_prj/package_out/sd_card.img.gz' into SD card, insert the SD card with the image into the destination board and power-on it.
After Linux booting on board, firstly install the Vitis AI Runtime (follow the steps in the document https://github.com/Xilinx/Vitis-AI/blob/master/setup/vck190/README.md) .
Then copy the folder '$TRD_HOME/app' in this TRD to the target folder "~/", and run below commands:

```
% cd ~/app/model/

% xdputil benchmark resnet50.xmodel -i -1 1

```
Expect result like below: 

```
I0617 10:35:50.797550  1962 test_dpu_runner_mt.cpp:473] shuffle results for batch...
I0617 10:35:50.799772  1962 performance_test.hpp:73] 0% ...
I0617 10:35:56.799910  1962 performance_test.hpp:76] 10% ...
.
.
.
I0617 10:36:50.801367  1962 performance_test.hpp:76] 100% ...
I0617 10:36:50.801432  1962 performance_test.hpp:79] stop and waiting for all threads terminated....
I0617 10:36:50.802978  1962 performance_test.hpp:85] thread-0 processes 75300 frames
I0617 10:36:50.803010  1962 performance_test.hpp:93] it takes 1564 us for shutdown
I0617 10:36:50.803024  1962 performance_test.hpp:94] FPS= 1254.93 number_of_frames= 75300 time= 60.0033 seconds.
I0617 10:36:50.803059  1962 performance_test.hpp:96] BYEBYE
```
###### **Note:** For running other network. Please refer to the [Vitis AI Github](https://github.com/Xilinx/Vitis-AI) and [Vitis AI User Guide](http://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf).


### 5.3 Change the Configuration

The DPUCVDX8G IP provides some user-configurable parameters, please refer to the document PG389 'Xilinx Versal DPU (DPUCVDX8G) Product Guide' .
In this TRD, user-configurable parameters are in the file '$TRD_HOME/vitis_prj/xvdpu_config.mk'. They are:
- CPB_N       -- number of AI Engine cores per batch handler, support 32 and 64.
- BATCH_N     -- number of batch engine integrated in DPUCVDX8G IP. Support 1 to 6 for CPB_N=32, and 1 to 5 for CPB_N=64.
- UBANK_IMG_N -- number of IMG BANKs are composed of UltraRAM, the Max is 16.
- UBANK_WGT_N -- number of WGT BANKs are composed of UltraRAM, the Max is 17.
- PL_FREQ     -- Frequency of 'm_axi_aclk', support range from 200M to 333M Hz. 

Other parameters are set by default as below:
- LOAD_PARALLEL_IMG   = 2
- SAVE_PARALLEL_IMG   = 2

After changing '$TRD_HOME/vitis_prj/xvdpu_config.mk', type 'make all' to build the design. 

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
- interrupt -- Interrupt signal generated by DPUCVDX8G.


DPUCVDX8G's connection with AIE and LPDDR's NOC are all defined in the '$TRD_HOME/vitis_prj/scripts/xvdpu_aie_noc.cfg' (generated by 'xvdpu_aie_noc.py').

For the clock design, please make sure that:
- s_axi_aclk for 's_axi_control' should use clock with lower frequency, such as 150M Hz, to get better timing.
- 'AI Engine Core Frequency' should be 4 times of DPUCVDX8G's m_axi_clk, or the maximum AIE frequency. In this TRD, it is 1250M Hz (the maximum AIE frequency of XCVC1902-2MP part on the VCK190 board). The value of 'AI Engine Core Frequency' can be set in the platform design files or '/vitis_prj/scripts/postlink.tcl'.

### 6.2 Changing Platform

Changing platform needs to modify 3 files: 'vitis_prj/Makefile', 'vitis_prj/scripts/xvdpu_aie_noc.py', and 'vitis_prj/scripts/postlink.tcl'.

**Note:** If the target platform is based on ES1 device, please check the known issue about the workaround for ES1 device.

1) 'vitis_prj/Makefile':
- Change the path of 'xpfm' file for varibale 'PLATFORM'
```
  PLATFORM           = */*.xpfm
```
- Change the path of 'rootfs.exts' and 'Image' in the package section (bottom of Makefile) 
```
  --package.rootfs     */rootfs.ext4 \
  --package.sd_file    */Image \
```

2) 'vitis_prj/scripts/xvdpu_aie_noc.py':
- Change the name of 'SP Tag'.

**Note:** The 'SP Tag' name of the slave AXI interfaces of LPDDR's NOC, is set in the platform design.
Take the default platform 'TRD platform1' as example, its 'SP Tag' are 'NOC_Sxx'. In 'vitis_prj/scripts/xvdpu_aie_noc.py', the name of 'SP Tag' is: "NOC_S" + str(S_AXI_N) . 

3) 'vitis_prj/scripts/postlink.tcl':

- Change the name of LPDDR's NOC
```
set cell_noc {*}
```
 ------

**Note:** Change setting of LPDDR to get the better performance of LPDDR, then better performance of DPUCVDX8G.
Some platform design may be confilict with below LPDDR setting. In this case, disable below line in 'vitis_prj/scripts/postlink.tcl' to get building step passed. But the performance of DPUCVDX8G may be affected.

```
set_property -dict [list CONFIG.MC_CHANNEL_INTERLEAVING {true} CONFIG.MC_CH_INTERLEAVING_SIZE {128_Bytes} CONFIG.MC_LPDDR4_REFRESH_TYPE {PER_BANK} CONFIG.MC_TRC {60000} CONFIG.MC_ADDR_BIT9 {CA5}] [get_bd_cells $cell_noc]
```
  
## 7 Basic Requirement of Platform
For platform which will integrate DPUCVDX8G, the basic requirements are listed as below:
- One 'CIPS' IP.
- One 'NOC' IP. Its slave AXI interfaces should be with 'SP Tag', and its DDR interface should provide best DDR bandwidth for DPUCVDX8G. For VCK190 board as example, the NOC should have 2x Memory Controller(for 2 LPDDR on board) and 4x MC ports. 
- One 'AI Engine' IP name 'ai_engine_0', and its Core Frequency should be 1250 MHz.
- One 'Clocking Wizard' IP, with at least 2 output clocks for DPUCVDX8G (150 MHz and 333 MHz).   
- Two 'Processor System Resets' IP, for 150 MHz and 333 MHz.
- One 'AXI smartconnect' IP with its master port enabled in the platform (for connection with DPUCVDX8G's 's_axi_control' port), and its slave interface connected to the CIPS master port. 
- One 'AXI interrupt controller' IP with its interrupt port connected to pl_ps_irqXX of CIPS and its slave AXI port connected to the CIPs master with its address mapped to 0xA5000000.    

For the detailed platform design, please refer to VCK190 platform in this TRD.

 ------
 
## 8 Vivado Project of TRD Platform1
Source files of VCK190 platform are in the folder '/vck190_platform/platforms'.

## 9 Known Issue 
1, Additional patch is needed for Psmnet supported by configuration CPB_N=64.
   
   Please follow 'https://github.com/Xilinx/Vitis-AI/blob/master/README.md#installing-patch-in-docker' to install the patch.

2, Workaround for ES1 device

For 2021.2 version platform based on ES1 device, before running apps, need firstly run workaround for ES1 device.

After board is booting up, create a script with below content, and run it on the ES1 board.


```
for i in {0..49}
do
  for j in {1..8}
  do
    a=0x20000000000
    b=0x31000
    devmem $[a+b+(i<<23)+(j<<18)] 32 0
  done
done
```