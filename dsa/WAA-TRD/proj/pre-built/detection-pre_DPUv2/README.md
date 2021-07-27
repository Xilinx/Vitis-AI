# Detection example: ZCU102 TRD run using Pre-processor source files & pre-built DPU IP

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

Download [Vitis-AI.1.4-WAA-TRD.bin.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Vitis-AI.1.4-WAA-TRD.bin.tar.gz). Untar the packet and copy `bin` folder to `Vitis-AI/dsa/WAA-TRD/`. 

The following tutorials assume that the Vitis and XRT environment variable is set as given below.

Open a linux terminal. Set the linux as Bash mode.

```
% source < vitis-install-directory >/Vitis/2020.2/settings64.sh
% source < path-to-XRT-installation-directory >/setup.sh
% gunzip < mpsoc-common-system >/xilinx-zynqmp-common-v2020.2/rootfs.tar.gz
% export EDGE_COMMON_SW=< mpsoc-common-system >/xilinx-zynqmp-common-v2020.2 
% export SDX_PLATFORM=< zcu102-base-platform-path >/xilinx_zcu102_base_202020_1/xilinx_zcu102_base_202020_1.xpfm

```

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

* Cross compile `adas_detection` example.
    ```
    cd  ~/Vitis-AI/dsa/WAA-TRD/app/adas_detection
    bash -x build.sh
    ``` 	
    If the compilation process does not report any error and the executable file `adas_detection` is generated, then the host environment is installed correctly.



## 2.5 Download Model files for Adas_detection

```
%	cd /Vitis-AI/dsa/WAA-TRD/app/adas_detection
%	mkdir model_zcu102
%	cd model_zcu102
%	wget https://www.xilinx.com/bin/public/openDownload?filename=yolov3_adas_pruned_0_9-zcu102_zcu104-r1.3.0.tar.gz -O yolov3_adas_pruned_0_9-zcu102_zcu104-r1.3.0.tar.gz
%	tar -xzvf yolov3_adas_pruned_0_9-zcu102_zcu104-r1.3.0.tar.gz
```


## 2.6 Run Adas detection Example
This part is about how to run the Adas detection example on zcu102 board.

Download the images at https://cocodataset.org/#download. Please select suitable images which has car, bicycle or pedestrian and copy these images to `Vitis-AI/dsa/WAA-TRD/app/adas_detection/data`. 


Copy the directory $TRD_HOME/app/adas_detection to the BOOT partition of the SD Card.

Please insert SD_CARD on the ZCU102 board.After the linux boot, run:

```
% cd /media/sd-mmcblk0p1/adas_detection
% cp /media/sd-mmcblk0p1/dpu.xclbin /usr/lib/
% export XILINX_XRT=/usr
% echo 1 > /proc/sys/kernel/printk
% mkdir output
% ./adas_detection model_zcu102/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 1

Expect:
Found Platform
Platform Name: Xilinx
INFO: Reading /usr/lib/dpu.xclbin
Loading: '/usr/lib/dpu.xclbin'
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0509 09:02:17.206205  1112 main.cc:458] create running for subgraph: subgraph_layer0-conv
Performance:18.2 FPS

```
