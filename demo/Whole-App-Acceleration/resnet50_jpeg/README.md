# Classification example:
:pushpin: **Note:** This application can be run only on ZCU102 platform.

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

## 2.2 Installing board image

* Download the SD card system image files from the following links:  

    [ZCU102](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-jpegppdpu-v2020.2-v1.4.0.img.gz)   

    Note: The version of the board image should be 2020.2 or above.
* Use Etcher software to burn the image file onto the SD card.
* Insert the SD card with the image into the destination board.
* Plug in the power and boot the board using the serial port to operate on the system.
* Set up the IP information of the board using the serial port.
	You can now operate on the board using SSH.

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
## 2.4 (Optional) Cross-compile WAA example
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

* Cross compile `resnet50_jpeg` example.
    ```
    cd  ~/Vitis-AI/demo/Whole-App-Acceleration/resnet50_jpeg
    bash -x build.sh
    ```
    If the compilation process does not report any error and the executable file `resnet50_jpeg` is generated , then the host environment is installed correctly.



## 2.5 Download Model files for Resnet50

```
%	cd /Vitis-AI/demo/Whole-App-Acceleration/resnet50_jpeg
%	mkdir model_zcu102
%	cd model_zcu102
%	wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102_zcu104-r1.3.0.tar.gz -O resnet50-zcu102_zcu104-r1.3.0.tar.gz
%	tar -xzvf resnet50-zcu102_zcu104-r1.3.0.tar.gz
```

## 2.6 Run Resnet50 Example
This part is about how to run the Resnet50 example on zcu102 board.

* Download the images at http://image-net.org/download-images and copy images to `Vitis-AI/demo/Whole-App-Acceleration/resnet50_jpeg/img` 

* Copy the directory Vitis-AI/demo/Whole-App-Acceleration/resnet50_jpeg to the BOOT partition of the SD Card.

* Please insert SD_CARD on the ZCU102 board. After the linux boot, run:

```
% cd /media/sd-mmcblk0p1/resnet50_jpeg
% cp /media/sd-mmcblk0p1/dpu.xclbin /usr/lib/
% export XILINX_XRT=/usr
% echo 1 > /proc/sys/kernel/printk
% #run with waa
% ./resnet50_jpeg model_zcu102/resnet50/resnet50.xmodel 1 1

Expect: 
number of images: 985
E2E Performance: 55.9302 fps

% #run without waa
% ./resnet50_jpeg model_zcu102/resnet50/resnet50.xmodel 0 1

Expect: 
number of images: 985
E2E Performance: 33.1453 fps

```
## 2.7 Performance
Below table shows the comparison of througput achieved by acclerating the pre-processing pipeline on FPGA. 
For `Resnet-50`, the performance numbers are achieved by running 1K images randomly picked from ImageNet dataset. 

Network: Resnet50
<table style="undefined;table-layout: fixed; width: 534px">
<colgroup>
<col style="width: 119px">
<col style="width: 136px">
<col style="width: 145px">
<col style="width: 134px">
</colgroup>
  <tr>
    <th rowspan="2">FPGA</th>
    <th colspan="2">E2E Throughput (fps)</th>
    <th rowspan="2"><span style="font-weight:bold">Percentage improvement in throughput</span></th>
  </tr>
  <tr>
    <td>with software Pre-processing</td>
    <td>with hardware Pre-processing</td>
  </tr>


  
  <tr>
   <td>ZCU102</td>
    <td>33.14</td>
    <td>55.93</td>
        <td>68.76%</td>
  </tr>

</table>


**Note that Performance numbers are computed using end-to-end latency and it depends on input image resolution. So performance numbers can vary with different images**  
