## ADAS detection
:pushpin: **Note:** This application can be run only on ZCU102 and Alveo-U50 platforms.

## Table of Contents

- [Introduction](#Introduction)
- [Setting Up and Running on ZCU102](#Setting-Up-and-Running-on-ZCU102)
    - [Setting Up the Target](#Setting-Up-the-Target-ZCU102)
    - [Building and running the application](#Building-and-running-the-application-on-ZCU102)
- [Setting Up and Running on U50](#Setting-Up-and-Running-on-Alveo-U50)
    - [Setting Up the Target](#Setting-Up-the-Target-Alveo-U50)
    - [Building and running the application](#Building-and-running-the-application-on-Alveo-U50)
- [Performance](#Performance)    

## Introduction

ADAS (Advanced Driver Assistance Systems) application
using YOLO-v3 network model is an example for object detection.
Accelerating pre-processing for YOLO-v3 is provided and can only run on ZCU102 board (device part xczu9eg-ffvb1156-2-e) and Alveo U50 card (device part xcu50-fsvh2104-2-e). In this application, software JPEG decoder is used for loading input image. Three processes are created one for image loading and running pre-processing kernel ,one for running the ML accelerator and one for generating output image. JPEG decoder transfer input image data to pre-processing kernel and the pre-processed data is transferred to the ML accelerator over a queue. ML accelerator output will be transferd over queue to create output image. Below image shows the inference pipeline.

<div align="center">
  <img width="75%" height="75%" src="../doc_images/block_dia_adasdetection.PNG">
</div>

## Setting Up and Running on ZCU102

### Setting Up the Target ZCU102

**To improve the user experience, the Vitis AI Runtime packages have been built into the board image. Therefore, user does not need to install Vitis AI
Runtime packages on the board separately.**

* Installing a Board Image.
	* Download the SD card system image files from the following links:  
	
      [ZCU102](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2020.2-v1.3.0.img.gz)   
	    
      	Note: The version of the board image should be 2020.2 or above.
	* Use Etcher software to burn the image file onto the SD card.
	* Insert the SD card with the image into the destination board.
	* Plug in the power and boot the board using the serial port to operate on the system.
	* Set up the IP information of the board using the serial port.
	You can now operate on the board using SSH.
	
* Update the system image files.
	* Download the [waa_system_zcu102_v1.4.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=waa_system_zcu102_v1.4.0.tar.gz).	

	* Copy the `waa_system_zcu102_v1.4.0.tar.gz` to the board using scp.
		```
		scp waa_system_zcu102_v1.4.0.tar.gz root@IP_OF_BOARD:~/
		```

	
	* Update the system image files on the target side

		```
		cd ~
		tar -xzvf waa_system_zcu102_v1.4.0.tar.gz
		cp waa_system_zcu102_v1.4.0/sd_card_adasdetection/* /mnt/sd-mmcblk0p1/
		cp /mnt/sd-mmcblk0p1/dpu.xclbin /usr/lib/
		ln -s /usr/lib/dpu.xclbin /mnt/dpu.xclbin
  		reboot
  		```
	  
* Download test images	

  For adas_detection example, download the images at https://cocodataset.org/#download and copy the images to `Vitis-AI/demo/Whole-App-Acceleration/adas_detection/data`

* Copy application files to SD card

    ```
	  scp -r Vitis-AI/demo/Whole-App-Acceleration/adas_detection root@IP_OF_BOARD:~/
    ```


### Building and running the application on ZCU102
* Build
    ```
      cd ~/adas_detection
      ./build.sh
      mkdir output #Will be written to the picture after processing
    ```
* Run adas_detection without waa
    ```
    ./adas_detection /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 0
    ```
* Run adas_detection with waa
    ```
    env XILINX_XRT=/usr ./adas_detection /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 1
    ```

## Setting Up and Running on Alveo U50
### Setting Up the Target Alveo U50
**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment**

* Follow the steps mentioned [here](../../../setup/alveo/README.md) to setup the target. 

* Download [waa_system_u50_v1.4.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=waa_system_u50_v1.4.0.tar.gz) and update the xclbin file.

  ```
	tar -xzvf waa_system_u50_v1.4.0.tar.gz
	sudo cp waa_system_u50_v1.4.0/detection/dpu.xclbin /usr/lib/
	sudo cp waa_system_u50_v1.4.0/detection/hbm_address_assignment.txt /usr/lib/
	```

* To download and install `adas detection` model:
	```
	  mkdir -p ${VAI_HOME}/demo/Whole-App-Acceleration/adas_detection/model
	  cd ${VAI_HOME}/demo/Whole-App-Acceleration/adas_detection/model
	  wget https://www.xilinx.com/bin/public/openDownload?filename=yolov3_adas_pruned_0_9-u50-r1.3.0.tar.gz -O yolov3_adas_pruned_0_9-u50-r1.3.0.tar.gz
	```	
* Install the model package.
	```
	  tar -xzvf yolov3_adas_pruned_0_9-u50-r1.3.0.tar.gz
	  sudo mkdir -p /usr/share/vitis_ai_library/models
	  sudo cp yolov3_adas_pruned_0_9 /usr/share/vitis_ai_library/models -r
	```
* Download test images	

  For adas_detection example, download the images at https://cocodataset.org/#download and copy the images to `Vitis-AI/demo/Whole-App-Acceleration/adas_detection/data`

### Building and running the application on Alveo U50
 *  Build
    ```
    cd ${VAI_HOME}/demo/Whole-App-Acceleration/adas_detection
    ./build.sh
    mkdir output #Will be written to the picture after processing
    ```
  * Run adas_detection without waa
    ```
    ./adas_detection /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 0
    ```
  * Run adas_detection with waa
    ```
    ./adas_detection /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 1
    ```

## Performance
Below table shows the comparison of througput achieved by acclerating the pre-processing pipeline on FPGA. 
For `Adas Detection`, the performance numbers are achieved by running 1K images randomly picked from COCO dataset. 

Network: YOLOv3 Adas Detection
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
    <td>11.1</td>
    <td>18.2</td>
        <td>63.9%</td>
  </tr>

  <tr>
   <td>U50</td>
    <td>29.6</td>
    <td>42.6</td>
        <td>43.9%</td>
  </tr>
</table>

**Note that Performance numbers are computed using end-to-end latency and it depends on input image resolution. So performance numbers can vary with different images**   
