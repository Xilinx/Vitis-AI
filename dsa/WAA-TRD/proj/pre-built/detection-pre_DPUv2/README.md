# Detection example: TRD run using Pre-processor source files & pre-built DPU IP

## 1 Software Tools and System Requirements

### Hardware

Required:

- ZCU102 evaluation board

- Micro-USB cable, connected to laptop or desktop for the terminal emulator

- SD card

### Software

  Required:
  - Vitis 2020.2[Vitis Core Development Kit](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/2020-2.html) 
  - [Silicon Labs quad CP210x USB-to-UART bridge driver](http://www.silabs.com/products/mcu/Pages/USBtoUARTBridgeVCPDrivers.aspx)
  - Serial terminal emulator e.g. [teraterm](http://logmett.com/tera-term-the-latest-version)
  - [XRT 2020.2](https://github.com/Xilinx/XRT/tree/2020.2)
  - [zcu102 base platform](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms.html)
  - [mpsoc common system](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zynqmp-common-v2020.1.tar.gz)


###### **Note:** The user can also refer the [zcu102 dpu platform](https://github.com/Xilinx/Vitis_Embedded_Platform_Source/tree/master/Xilinx_Official_Platforms/zcu102_dpu), The github page includes all the details, such as how to generage the zcu102 dpu platform, how to create the SD card after compiling the DPU project.
------


## 2 Tutorial

### 2.1 Board Setup

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


### 2.2 Build and run the application

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
Note that **mpsoc common system** should be downloaded in the 1 chapter. 



#### Generate SD card image

```
% cd $TRD_HOME/proj/pre-built/detection-pre_DPUv2
% ./run.sh 
```
Note that 

- Generated SD card image will be here **$TRD_HOME/proj/pre-built/detection-pre_DPUv2/binary_container_1/sd_card.img**.
- Pre-built DPU IP is configured to **B4096**.
- Build runtime is ~30 min.

### 2.2 Installing board image
- Use Etcher software to burn the sd card image file onto the SD card.

## 2.3 Installing Vitis AI Runtime on the Evaluation Board

- Download the [Vitis AI Runtime 1.3.0](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.3.0.tar.gz)  
	
- Untar the runtime packet and copy the following folder to the board using scp.
```
	tar -xzvf vitis-ai-runtime-1.3.0.tar.gz
	scp -r vitis-ai-runtime-1.3.0/aarch64/centos root@IP_OF_BOARD:~/
```
- Log in to the board using ssh. You can also use the serial port to login.
- Install the Vitis AI Runtime. Execute the following command in order.
```
	cd ~/centos
    rpm2cpio libvart-1.3.0-r<x>.aarch64 | cpio -idmv
	rpm -ivh --force libunilog-1.3.0-r<x>.aarch64.rpm
	rpm -ivh --force libxir-1.3.0-r<x>.aarch64.rpm
	rpm -ivh --force libtarget-factory-1.3.0-r<x>.aarch64.rpm
	rpm -ivh --force libvart-1.3.0-r<x>.aarch64.rpm
	rpm -ivh --force libvitis_ai_library-1.3.0-r<x>.aarch64.rpm
```
## 2.4 Run Adas detection Example
This part is about how to run the Adas detection example on zcu102 board.

Download the images at https://cocodataset.org/#download. Please select suitable images which has car, bicycle or pedestrian and copy these images to `Vitis-AI/dsa/WAA-TRD/app/adas_detection_waa/data`. 

Copy the directory $TRD_HOME/app/adas_detection_waa to the BOOT partition of the SD Card.

Please insert SD_CARD on the ZCU102 board.After the linux boot, run:

```
% cd /mnt/sd-mmcblk0p1/adas_detection_waa
% export XILINX_XRT=/usr
% cp /mnt/sd-mmcblk0p1/dpu.xclbin /usr/lib/
% mkdir output
% ./adas_detection_waa model/yolov3_adas_pruned_0_9.xmodel

Expect: 
Input Image:./data/<img>.jpg
Output Image:./output/<img>.jpg

```