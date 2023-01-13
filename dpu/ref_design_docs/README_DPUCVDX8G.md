<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# VCK190 DPUCVDX8G reference design for Vitis AI

**Note:** This Technical Reference Design (TRD) is based on the *production* version of the VCK190 evaluation board.

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
		- [5.2.1 Build the DPU](#521-build-the-dpu)
		- [5.2.2 Get Json File](#522-get-json-file)
		- [5.2.3 Run ResNet50 Example](#523-run-resnet50-example)
	- [5.3 Change the Configuration](#53-change-the-configuration)
		- [5.3.1 Single CU](#531-single-cu)
		- [5.3.2 Multiple CUs](#532-multiple-cus)
		- [5.3.3 PSMNet](#533-psmnet)
- [6 Instructions for Changing the Platform](#6-instructions-for-changing-the-platform)
	- [6.1 DPUCVDX8G Ports](#61-dpucvdx8g-ports)
	- [6.2 Changing the Platform](#62-changing-the-platform)
- [7 Basic Requirement of Platform](#7-basic-requirement-of-platform)
- [8 Vivado Project of TRD Platform](#8-vivado-project-of-trd-platform)
- [9 Instructions for Adding Other Kernels](#9-instructions-for-adding-other-kernels)
	- [9.1 RTL Kernel](#91-rtl-kernel)
	- [9.2 AI Engine Kernel](#92-ai-engine-kernel)
	- [9.3 Kernel with RTL and AI Engine](#93-kernel-with-rtl-and-ai-engine)
- [10 Examples of MIPI Usage](#10-examples-of-mipi-usage)
- [11 Known Issues ](#11-known-issues)

## 1 Revision History

Vitis3.0 Change log:
- Change platform as VCK190-prod 2022.2 version
- Update DPUCVDX8G IP to support 'Correlation 1D and 2D', and 'Argmax and Max along channel dimension'

Vitis2.5 Change log:
- Change platform as VCK190-prod 2022.1 version
- Add DPUCVDX8G configuration： Multiple CUs, PSMNet

Vitis2.0 Change log:
- Change platform as VCK190-prod 2021.2 version
- Change AI Engine Core Frequency from 1333 MHz to 1250 MHz
- Add DPUCVDX8G configurations： C64B1，C64B2，C64B4，C64B5. All supported configurations are C32B[1:6] and C64B[1:5]

Vitis1.4.1 Change log:
- Change platform as VCK190-prod 2021.1 version
- Update scripts to make changing platform easier, update 'vitis_prj/Makefile' to simplify the HW build steps
- Add DPUCVDX8G configuration C64B3
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

- Production version of the VCK190 evaluation board

- USB type-C cable, connected to a PC for the terminal emulator

- SD card 

### 3.2 Software

  Required:
  - Vitis 2022.2
  - Python (version 2.7.5 or 3.6.8)
  - csh
###### **Note:** `bash` is used during the build but some `csh` scripts are used.
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
    ├── psmnet
    └── rtl

```
------

## 5 Tutorials

### 5.1 Board Setup

###### Board jumper and switch settings:

Make sure the board is set as booting from SD card (see [UG1366]):

- Remove J326 (7-8) jumper.


- SW11[4:1]- [OFF,OFF,OFF,ON].


- SW1[4:1]- [OFF,OFF,OFF,ON].

### 5.2 Build and Run TRD Flow

The following tutorials assume that the '$TRD_HOME' environment variable is set as shown below.

```
% export TRD_HOME =<Vitis AI path>/reference_design/DPUCVDX8G-TRD
```

**Step1:** Build VCK190 platform 

Firstly, build the VCK190 platform in the folder '$TRD_HOME/vck190_platform', following the instructions in '$TRD_HOME/vck190_platform/README.md".


**Step2:** Setup environment for building DPUCVDX8G

When platform is ready, set the Vitis environment variable as given below.

Open a linux terminal. Set the linux as Bash mode.

```
% source <vitis install path>/Vitis/2022.2/settings64.sh

```
###### **Note:** Some NOC parameters in Vitis 2022.2 have been changed, which will affect DDR performance. To get performance back, please add the workaround in the section [10 Known Issue].

#### 5.2.1 Build the DPU

The default architecture of DPUCVDX8G is C32B3CU1 (CU_N=1, CPB_N=32, BATCH_SingleCU=3), PL clock frequency is 333 MHz, UBANK_IMG_N=16 and UBANK_WGT_N=17.
Modify file '$TRD_HOME/vitis_prj/xvdpu_config.mk' can change the parameters.

Add the DPU to the platform.

```
% cd $TRD_HOME/vitis_prj

% make all

```

The generated SD card image:  $TRD_HOME/vitis_prj/package_out/sd_card.img.gz

The implemented Vivado project: $TRD_HOME/vitis_prj/hw/binary_container_1/link/vivado/vpl/prj/prj.xpr

**Note1:** With 'make help' to see more detailed information about the commands. 

**Note2:** The implementation strategy is set in the file '$TRD_HOME/vitis_prj/scripts/system.cfg', the default strategy is " prop=run.impl_1.strategy=Performance_ExploreWithRemap ".
           The implementation strategy may be changed by editing `system.cfg`.  

**Note3:** With same configuration of DPUCVDX8G, 'libadf.a' file of AI Engine can be reused. Comment out the last line of '$TRD_HOME/vitis_prj/Makefile', to save compile time for re-building the hardware design.
```
# -@rm -rf aie
```
		   
#### 5.2.2 Get Json File

The 'arch.json' file is an important file required by Vitis AI. It works together with Vitis AI compiler to support model compilation with various DPUCVDX8G configurations. The 'arch.json' file will be generated by Vitis during the compilation of DPUCVDX8G TRD, it can be found in '$TRD_HOME/vitis_prj/package_out/sd_card' .

It can also be found in the following path:

$TRD_HOME/vitis_prj/hw/binary_container_1/link/vivado/vpl/prj/prj.gen/sources_1/bd/vck190*/ip/*_DPUCVDX8G_1_0/arch.json


#### 5.2.3 Run ResNet50 Example

The TRD project has generated the matching model file in '$TRD_HOME/app' path for the default settings. If the settings of DPUCVDX8G are modified, the model needs to be created according to the fingerprint in 'arch.json'.

This section is about how to run the ResNet50 example.

Use the balenaEtcher tool to flash '$TRD_HOME/vitis_prj/package_out/sd_card.img.gz' into SD card, insert the SD card with the image into the destination board and power-on it.
After Linux booting on board, firstly install the Vitis AI Runtime (follow the steps in the document https://github.com/Xilinx/Vitis-AI/blob/master/setup/vck190/README.md) .
Then copy the folder '$TRD_HOME/app' in this TRD to the target folder "~/", and run the following commands:

```
% cd ~/app/model/

% xdputil benchmark resnet50.xmodel -i -1 1

```
A typical output would appear as shown below:

```
I1123 04:08:22.475286  1127 test_dpu_runner_mt.cpp:474] shuffle results for batch...
I1123 04:08:22.476413  1127 performance_test.hpp:73] 0% ...
I1123 04:08:28.476716  1127 performance_test.hpp:76] 10% ...
.
.
.
I1123 04:09:22.478189  1127 performance_test.hpp:76] 100% ...
I1123 04:09:22.478253  1127 performance_test.hpp:79] stop and waiting for all threads terminated....
I1123 04:09:22.478495  1127 performance_test.hpp:85] thread-0 processes 81906 frames
I1123 04:09:22.478528  1127 performance_test.hpp:93] it takes 254 us for shutdown
I1123 04:09:22.478543  1127 performance_test.hpp:94] FPS= 1365.05 number_of_frames= 81906 time= 60.0021 seconds.
I1123 04:09:22.478579  1127 performance_test.hpp:96] BYEBYE

```
###### **Note:** For running other networks, refer to the [Vitis AI Github](https://github.com/Xilinx/Vitis-AI) and [Vitis AI User Guide](http://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf).


### 5.3 Change the Configuration

The DPUCVDX8G IP provides some user-configurable parameters, refer to the document PG389 'Xilinx Versal DPU (DPUCVDX8G) Product Guide' .
In this TRD, user-configurable parameters are in the file '$TRD_HOME/vitis_prj/xvdpu_config.mk'. They are:
- CU_N        -- Compute Unit (CU) number (maximum value is 3).
- CPB_N       -- number of AI Engine cores per batch handler (32 or 64).
- BATCH_SingleCU -- number of batch engine integrated in DPUCVDX8G IP for CU_N=1. Support 1 to 6 for CPB_N=32, and 1 to 5 for CPB_N=64.
- UBANK_IMG_N -- number of IMG BANKs are composed of UltraRAM (maximum value is 16).
- UBANK_WGT_N -- number of WGT BANKs are composed of UltraRAM (maximum value is 17).
- PL_FREQ     -- Frequency of 'm_axi_aclk' (200MHz ~ 333MHz). 
- PSMNET_EN   -- Default is '0'. When set as '1', will enable additional kernels for running PSMNet, together with DPUCVDX8G (C32B3CU1).

###### **Note:** When CU_N =2 or CU_N=3, BATCH_N is fixed as 1. When CU_N=1, BATCH_N=BATCH_SingleCU. 

Other parameters are set by default as below:
- LOAD_PARALLEL_IMG   = 2
- SAVE_PARALLEL_IMG   = 2

After changing '$TRD_HOME/vitis_prj/xvdpu_config.mk', type 'make all' to build the design. 

#### 5.3.1 Single CU

For single CU(Compute Unit) configuration, 'CU_N' is fixed as 1. Below 3 parameters must be specified.

```
- CU_N  = 1 
- CPB_N = 32, or 64
- BATCH_SingleCU = 1,2,3,4,5, or 6 ( When CPB_N=64, the Max is 5)
```

#### 5.3.2 Multiple CUs

For multiple CUs configuration, 'CU_N' can be 2 or 3, 'BATCH_N' is fixed as 1. The parameter 'BATCH_SingleCU' will be ignored.

Below 2 paraments must be specified:
```
- CU_N  = 2 or 3
- CPB_N = 32 or 64
```

#### 5.3.3 PSMNet

For PSMNet, set 'PSMNET_EN  = 1'. It will add 3 additional Vitis kernels to work together with DPUCVDX8G (C32B3CU1).
- mm2s (RTL kernel)
- ss2m (RTL Kernel)
- psmnet_adf (AI Engine Kernel)

------

## 6 Instructions for Changing the Platform

### 6.1 DPUCVDX8G Ports

The DPUCVDX8G ports are listed as below.

- m*_wgt/img/bias/instr_axi -- Master AXI interfaces, connected with LPDDR's NOC (NOC_0 in this TRD platform)
- s*_ofm_axis     -- Slave AXI-stream interface, connected with AI Engine (ai_engine_0).
- m*_ifm/wgt_axis -- Master AXI-stream interface, connected with AI Engine (ai_engine_0).
- m_axi_clk   -- Input clock used for DPUCVDX8G general logic, AXI and AXI-stream interface. Frequency is 333M Hz in this TRD.
- m_axi_aresetn -- Active-Low reset for DPUCVDX8G general logic. 
- s_axi_control  -- AXI lite interface for controlling DPUCVDX8G registers, connected with CIPs through AXI_Smartconnect_IP.
- s_axi_aclk   -- Input clock for S_AXI_CONTROL. Frequency is 150M Hz in this TRD.
- s_axi_aresetn -- Active-Low reset for S_AXI_CONTROL.
- interrupt -- Interrupt signal generated by DPUCVDX8G.


DPUCVDX8G's connection with AI Engine array and LPDDR's NOC are all defined in the '$TRD_HOME/vitis_prj/scripts/xvdpu_aie_noc.cfg' (generated by 'xvdpu_aie_noc.py').

For the clock design, make sure that:
- s_axi_aclk for 's_axi_control' should use clock with lower frequency (e.g. 150M Hz) to get better timing.
- 'AI Engine Core Frequency' should be 4 times of DPUCVDX8G's m_axi_clk, or the maximum AI Engine frequency. In this TRD, it is 1250M Hz (the maximum AI Engine frequency of XCVC1902-2MP device on the VCK190 board). The value of 'AI Engine Core Frequency' can be set in the platform design files or '/vitis_prj/scripts/postlink.tcl'.

### 6.2 Changing the Platform

Changing platform needs to modify 3 files: 'vitis_prj/Makefile', 'vitis_prj/scripts/xvdpu_aie_noc.py', and 'vitis_prj/scripts/postlink.tcl'.

**Note:** If the target platform is based on ES1 device, check the known issue about the workaround for ES1 device.

1) 'vitis_prj/Makefile':
- Change the path of 'xpfm' file for the varibale 'PLATFORM'
```
  PLATFORM           = */*.xpfm
```
- Change the path of 'rootfs.exts' and 'Image' in the package section (at the bottom of Makefile) 
```
  --package.rootfs     */rootfs.ext4 \
  --package.sd_file    */Image \
```

2) 'vitis_prj/scripts/xvdpu_aie_noc.py':
- Change the name of 'SP Tag'.

**Note:** The 'SP Tag' name of the slave AXI interfaces of LPDDR's NOC, is set in the platform design.
Take the default platform 'TRD platform' as example, its 'SP Tag' is 'NOC_Sxx'. In 'vitis_prj/scripts/xvdpu_aie_noc.py', the name of 'SP Tag' is: "NOC_S" + str(S_AXI_N) . 

3) 'vitis_prj/scripts/postlink.tcl':

- Change the name of LPDDR's NOC
```
set cell_noc {*}
```
 ------

**Note:** Change setting of LPDDR to get the better performance of LPDDR, then better performance of DPUCVDX8G.
Some platform design may be confilict with below LPDDR setting. In this case, disable below line in 'vitis_prj/scripts/postlink.tcl' to get building step passed. But the performance of DPUCVDX8G may be degraded.

```
set_property -dict [list CONFIG.MC_CHANNEL_INTERLEAVING {true} CONFIG.MC_CH_INTERLEAVING_SIZE {128_Bytes} CONFIG.MC_LPDDR4_REFRESH_TYPE {PER_BANK} CONFIG.MC_TRC {60000} CONFIG.MC_ADDR_BIT9 {CA5}] [get_bd_cells $cell_noc]
```
  
## 7 Basic Requirement of Platform
For platform which will integrate DPUCVDX8G, the basic requirements are listed below:
- One 'CIPS' IP.
- One 'NOC' IP. Its slave AXI interfaces should have 'SP Tag', and its DDR interface should provide the best DDR bandwidth for DPUCVDX8G. For VCK190 board, the NOC should have 2x Memory Controller(for 2 LPDDR on board) and 4x MC ports. 
- One 'AI Engine' IP name 'ai_engine_0', and its Core Frequency should be up to the maximum supported by the device.
- One 'Clocking Wizard' IP, with at least 2 output clocks for DPUCVDX8G (150 MHz and 333 MHz).   
- Two 'Processor System Resets' IP, for 150 MHz and 333 MHz.
- One 'AXI smartconnect' IP with its master port enabled in the platform (for connection with DPUCVDX8G's 's_axi_control' port), and its slave interface connected to the CIPS master port. 
- One 'AXI interrupt controller' IP with its interrupt port connected to pl_ps_irqXX of CIPS and its slave AXI port connected to the CIPs master with its address mapped to 0xA5000000.    

For the detailed platform design, refer to VCK190 platform in this TRD.

 ------
 
## 8 Vivado Project of TRD Platform
Source files of the VCK190 platform are in the folder '/vck190_platform/platforms'.

## 9 Instructions for Adding Other Kernels

Vitis kernels developed for Versal devices, could be RTL kernel (only use PL resouces), AIE kernel (only uses AI Engine tiles), or kernel including both PL and AIE. The basic instructions for adding other kernels in this TRD are shown below.


### 9.1 RTL Kernel

Package the RTL kernel as XO file. Then modify 2 files: 'vitis_prj/Makefile', and 'vitis_prj/scripts/xvdpu_aie_noc.py',

1) 'vitis_prj/Makefile': 
- Add the name of XO files in the parameters 'BINARY_CONTAINER_1_OBJS' by adding following command
```
BINARY_CONTAINER_1_OBJS   += xxx.xo
```
- In the v++ linking command line, specify the clock frequency for the clock soure of RTL kernel.
```
--clock.freqHz <freqHz>:<kernelName.clk_name>
```
2) 'vitis_prj/scripts/xvdpu_aie_noc.py':
- Create instance for the RTL kernel, and map kernel ports to memory (LPDDR's NOC)
```
result += "nk=<kernel name>:<number>:<cu_name>.<cu_name>...\n" 
result += "sp=<cu_name>.<port>:<SP Tag of NOC's slave port>\n"
```

Refer to the detailed commands for adding 2 RTL kernel 'mm2s' and 'ss2m' in PSMNet configuration.
For the platform in this TRD, the 'SP Tag of NOC's slave port' are 'NOC_Sxx'. Use the 'SP Tag' which are not used by DPUCVDX8G.
Take PSMNet configuration in this TRD as example, DPUCVDX8G will use NOC_S21~S38, then 'mm2s' and 'ss2m' will use NOC_S39 and NOC_S40.

```
sp=mm2s_1.m_axi_gmem:NOC_S39
sp=s2mm_1.m_axi_gmem:NOC_S40
```

### 9.2 AI Engine Kernel

Multiple AI Engine (AIE) kernels are combined in an adaptive data flow (ADF) graph. Graph of each AIE kernels will be as sub-graph of the ADF graph. The aiecompiler will compile the ADF graph as 'libadf.a', which will be linked with platform using v++ command.

Need to modify 3 files: 'xvdpu_ip/aie/graph_comb.cc', 'xvdpu_ip/aie/scripts/compile_ip.csh.xxx', and 'vitis_prj/scripts/xvdpu_aie_noc.py'
1) 'xvdpu_ip/aie/graph_comb.cc':
- Add the sub-graph of AIE kernel by inserting below line
```
#include "<graph of AIE kernel>"
```
Refer to 'xvdpu_ip/psmnet/graph_comb.cc' for adding sub-graph.

**Note:** For PSMNet IP, it will be copied to 'vitis_prj/aie/psmnet' by 'vitis_prj/scripts/makefile'. 

2) 'xvdpu_ip/aie/scripts/compile_ip.csh.xxx':
- Add the source code folder of AI Engine kernel
```
 -include="<source code folder of AI Engine kernel>" \
```
Refer to 'xvdpu_ip/psmnet/compile_ip.csh' for adding the source code folder of AI Engine kernel.

3) 'vitis_prj/scripts/xvdpu_aie_noc.py':

- If GMIOs are used by AI Engine kernel to connect with memory (LPDDR's NOC), need to change the SP Tag used by DPUCVDX8G. This can be implemented by modifying the parameter 'S_AXI_N'.

```
S_AXI_N = xx

```
Take PSMNet as example, its AI Engine kernels will use 8 GMIOs, so need to change S_AXI_N = 21.

**Note:** In the platform in this TRD, SP Tag of LPDDR's NOC can be used by kernels is started from NOC_S13.

### 9.3 Kernel with RTL and AI Engine

For kernel with both RTL and AI Engine, the AXI-stream connections between its RTL and AI Engine are also need to be specified.

Except the mdifications in section 9.1 and 9.2, one more modification needs to be done in 'vitis_prj/scripts/xvdpu_aie_noc.py'.
Add below lines in 'xvdpu_aie_noc.py'.

```
result += "stream_connect=<master port of RTL>:ai_engine_0.Sxx_AXIS\n"
result += "stream_connect=ai_engine_0.Mxx_AXIS:<slave port of RTL>\n"
```

The detailed AXI-stream connection between RTL and AI Engine can be found in the file 'vitis_prj/aie/ip/Work/arch/cfgraph.xml'. The 'cfgraph.xml' is generated by aiecompiler. To get 'cfgraph.xml', firstly do the modifications in the scection 9.2, then run command 'make files' and 'make aie'. 

## 10 Examples of MIPI Usage
1, Capture from MIPI Sensor, Output to HDMI Monitor

```
/etc/init.d/xserver-nodm stop
gst-launch-1.0 mediasrcbin media-device=/dev/media1 ! 'video/x-raw, width=1920, height=1080, format=YUY2, framerate=60/1' ! perf ! kmssink driver-name=xlnx plane-id=41 sync=false
```

2, How to set the MIPI Sensor if the display is unclear or dark

```
v4l2-ctl -d /dev/v4l-subdev0 -c exposure=10000
# rerun
gst-launch-1.0 mediasrcbin media-device=/dev/media1 ! 'video/x-raw, width=1920, height=1080, format=YUY2, framerate=60/1' ! perf ! kmssink driver-name=xlnx plane-id=41 sync=false
```

3, How to view the range of Values

```
root@xilinx-vck190-20222:~# v4l2-ctl --list-ctrls --device /dev/v4l-subdev0
User Controls
                       exposure 0x00980911 (int)    : min=14 max=16666 step=1 default=14 value=14
                           gain 0x00980913 (int)    : min=256 max=46088 step=1 default=5120 value=5120
                  vertical_flip 0x00980915 (bool)   : default=0 value=0
Image Processing Controls
                   test_pattern 0x009f0903 (menu)   : min=0 max=12 default=0 value=0

```

## 11 Known Issues 
1, DDR performance drop in Vitis 2022.2

Some NOC parameters in Vitis 2022.2 have been changed, which will affect DDR performance, and consequently reduce DPUCVDX8G throughput. 

To get performance back, there is workaround to change the default NOC parameters for Vitis 2022.2.

Adding the following line to your tcl scripts '$HOME/.Xilinx/Vivado/Vivado_init.tcl'

```
set_param place.preplaceNOC true

```

For details about 'Vivado_init.tcl', refer to the link page 'https://docs.xilinx.com/r/en-US/ug894-vivado-tcl-scripting/Initializing-Tcl-Scripts'.


2, Workaround for ES1 Device

For platform based on an ES1 device, before running apps, need firstly run workaround for ES1 device.

After board is booting up, create a script with the following content, and run it on the ES1 board.


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

<!--                                                                      
* Copyright 2019 Xilinx Inc.                                               
*                                                                          
* Licensed under the Apache License, Version 2.0 (the "License");          
* you may not use this file except in compliance with the License.         
* You may obtain a copy of the License at                                  
*                                                                          
*    http://www.apache.org/licenses/LICENSE-2.0                            
*                                                                          
* Unless required by applicable law or agreed to in writing, software      
* distributed under the License is distributed on an "AS IS" BASIS,        
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
* See the License for the specific language governing permissions and      
* limitations under the License.                                           
-->
