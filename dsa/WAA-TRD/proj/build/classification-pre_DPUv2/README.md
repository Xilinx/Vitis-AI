# Classification example: ZCU102 TRD run using Pre-processor & DPU source files

## 1 Software Tools and System Requirements

### Hardware

Required:

- ZCU102 evaluation board

- Micro-USB cable, connected to laptop or desktop for the terminal emulator

- SD card

### Software

  Required:
  - Vitis 2020.2[Vitis Core Development Kit](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/2020-2.html) 
  - [CP210x_Universal_Windows_Driver](https://www.silabs.com/documents/public/software/CP210x_Universal_Windows_Driver.zip)
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
%export TRD_HOME =< Vitis-AI-path >/dsa/WAA-TRD
```

###### **Note:** It is recommended to follow the build steps in sequence.

We need install the Vitis Core Development Environment.

Download and unzip mpsoc common system & zcu102 base platform package from chapter 1.

The following tutorials assume that the Vitis and XRT environment variable is set as given below.

Open a linux terminal. Set the linux as Bash mode.

```
% source < vitis-install-directory >/Vitis/2020.2/settings64.sh
% source < path-to-XRT-installation-directory >/setup.sh
% gunzip < mpsoc-common-system >/xilinx-zynqmp-common-v2020.2/rootfs.tar.gz
% export EDGE_COMMON_SW=< mpsoc-common-system >/xilinx-zynqmp-common-v2020.2 
% export SDX_PLATFORM=< zcu102-base-platform-path >/xilinx_zcu102_base_202020_1/xilinx_zcu102_base_202020_1.xpfm

```

### Generate SD card image

```
% cd $TRD_HOME/proj/build/classification-pre_DPUv2
% ./build_classification_pre.sh
```
Note that 
- Generated SD card image will be here **$TRD_HOME/proj/build/classification-pre_DPUv2/binary_container_1/sd_card.img**.
- The default setting of DPU is **B4096** with RAM_USAGE_LOW, CHANNEL_AUGMENTATION_ENABLE, DWCV_ENABLE, POOL_AVG_ENABLE, RELU_LEAKYRELU_RELU6, Softmax. Modify the `$TRD_HOME/proj/build/classification-pre_DPUv2/dpu_conf.vh` file can change the default settings.
- Build runtime is ~4.5 hours

## 2.2 Installing board image
- Use Etcher software to burn the sd card image file onto the SD card.


## 2.3 Installing Vitis AI Runtime on the Evaluation Board

- Download the [Vitis AI Runtime 1.4.0](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.4.0.tar.gz). 

- Untar the runtime packet and copy the following folder to the board using scp.
```
	tar -xzvf vitis-ai-runtime-1.4.0.tar.gz
	scp -r vitis-ai-runtime-1.4.0/2020.2/aarch64/centos root@IP_OF_BOARD:~/
```
- Install the Vitis AI Runtime on the evaluation board. Execute the following command.
```
	cd ~/centos
	bash setup.sh
```
## 2.4 (Optional) Cross-compile WAA-TRD example
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

* Cross compile `resnet50` example.
    ```
    cd  ~/Vitis-AI/dsa/WAA-TRD/app/resnet50
    bash -x build.sh
    ```
    If the compilation process does not report any error and the executable file `resnet50` is generated , then the host environment is installed correctly.



## 2.5 Download Model files for Resnet50

```
%	cd /Vitis-AI/dsa/WAA-TRD/app/resnet50
%	mkdir model_zcu102
%	cd model_zcu102
%	wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102_zcu104-r1.3.0.tar.gz -O resnet50-zcu102_zcu104-r1.3.0.tar.gz
%	tar -xzvf resnet50-zcu102_zcu104-r1.3.0.tar.gz
```

## 2.6 Run Resnet50 Example
This part is about how to run the Resnet50 example on zcu102 board.

* Download the images at http://image-net.org/download-images and copy images to `Vitis-AI/dsa/WAA-TRD/app/resnet50/img` 

* Copy the directory $TRD_HOME/app/resnet50 to the BOOT partition of the SD Card.

* Please insert SD_CARD on the ZCU102 board. After the linux boot, run:

```
% cd /media/sd-mmcblk0p1/resnet50
% cp /media/sd-mmcblk0p1/dpu.xclbin /usr/lib/
% export XILINX_XRT=/usr
% echo 1 > /proc/sys/kernel/printk
%
% #run with waa
%./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel 1 0

Expect:
Image : ./img/bellpeppe-994958.JPEG
top[0] prob = 0.990457  name = bell pepper
top[1] prob = 0.004048  name = acorn squash
top[2] prob = 0.002455  name = cucumber, cuke
top[3] prob = 0.000903  name = zucchini, courgette
top[4] prob = 0.000703  name = strawberry

``

% #run without waa
%./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel 0 0

Expect:
Image : ./img/bellpeppe-994958.JPEG
top[0] prob = 0.992920  name = bell pepper
top[1] prob = 0.003160  name = strawberry
top[2] prob = 0.001493  name = cucumber, cuke
top[3] prob = 0.000705  name = acorn squash
top[4] prob = 0.000428  name = zucchini, courgette`
