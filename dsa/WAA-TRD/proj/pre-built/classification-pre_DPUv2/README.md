# Classification example: TRD run using Pre-processor files & pre-built DPU

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
  - [zcu102 base platform](https://www.xilinx.com/member/forms/download/design-license-zcu102-base.html?filename=xilinx_zcu102_base_202020_1.zip)
  - [mpsoc common system](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zynqmp-common-v2020.2.tar.gz)

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
%export TRD_HOME =< Vitis-AI-path >/WAA-TRD
```

###### **Note:** It is recommended to follow the build steps in sequence.

We need install the Vitis Core Development Environment.

Download and unzip mpsoc common system & zcu102 base platform package from chapter 1.

Download [Vitis-AI.1.3-WAA-TRD.bin.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Vitis-AI.1.3-WAA-TRD.bin.tar.gz). Untar the packet and copy `bin` folder to `Vitis-AI/dsa/WAA-TRD/`. 

The following tutorials assume that the Vitis and XRT environment variable is set as given below.

Open a linux terminal. Set the linux as Bash mode.

```
% source < vitis-install-directory >/Vitis/2020.2/settings64.sh
% source < part-to-XRT-installation-directory >/setup.sh
% gunzip < mpsoc-common-system >/xilinx-zynqmp-common-v2020.2/rootfs.tar.gz
% export EDGE_COMMON_SW=< mpsoc-common-system >/xilinx-zynqmp-common-v2020.2 
% export SDX_PLATFORM=< zcu102-base-platform-path >/xilinx_zcu102_base_202020_1/xilinx_zcu102_base_202020_1.xpfm

```

### Generate SD card image

```
% cd $TRD_HOME/proj/pre-built/classification-pre_DPUv2
% ./run.sh
```
Note that 
- Generated SD card image will be here **$TRD_HOME/proj/pre-built/classification-pre_DPUv2/binary_container_1/sd_card.img**.
- Pre-built DPU IP is configured to **B4096**.
- Build runtime is ~30 min.

## 2.2 Installing board image
- Use Etcher software to burn the sd card image file onto the SD card.


## 2.3 Installing Vitis AI Runtime on the Evaluation Board

- Download the  [Vitis AI Runtime 1.3.0](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.3.0.tar.gz). 
	
- Untar the runtime packet and copy the following folder to the board using scp.
```
	tar -xzvf vitis-ai-runtime-1.3.0.tar.gz
	scp -r vitis-ai-runtime-1.3.0/aarch64/centos root@IP_OF_BOARD:~/
```
- Install the Vitis AI Runtime on the evaluation board. Execute the following command.
```
	cd ~/centos
	bash setup.sh
```

## 2.4 Download Model files for Resnet50

```
%	cd /Vitis-AI/dsa/WAA-TRD/app/resnet50_waa
%	mkdir model_zcu102
%	cd model_zcu102
%	wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102_zcu104-r1.3.0.tar.gz -O resnet50-zcu102_zcu104-r1.3.0.tar.gz
%	tar -xzvf resnet50-zcu102_zcu104-r1.3.0.tar.gz
```

## 2.5 Run Resnet50 Example
This part is about how to run the Resnet50 example on zcu102 board.

* Download the images at http://image-net.org/download-images and copy images to `Vitis-AI/dsa/WAA-TRD/app/resnet50_waa/img` 

* Copy the directory $TRD_HOME/app/resnet50_waa to the BOOT partition of the SD Card.

* Please insert SD_CARD on the ZCU102 board. After the linux boot, run:

```
% cd /media/sd-mmcblk0p1/resnet50_waa
% cp /media/sd-mmcblk0p1/dpu.xclbin /usr/lib/
% export XILINX_XRT=/usr
% echo 1 > /proc/sys/kernel/printk
% ./resnet50_waa model_zcu102/resnet50/resnet50.xmodel

Expect: 
Image : ./img/bellpeppe-994958.JPEG
top[0] prob = 0.990457  name = bell pepper
top[1] prob = 0.004048  name = acorn squash
top[2] prob = 0.002455  name = cucumber, cuke
top[3] prob = 0.000903  name = zucchini, courgette
top[4] prob = 0.000703  name = strawberry

```
