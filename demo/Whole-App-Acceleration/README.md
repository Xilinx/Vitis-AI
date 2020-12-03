# Whole Application Acceleration: Accelerating ML Preprocessing for Classification and Detection networks

## Introduction

This application demonstrates how Xilinx® [Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) functions can be integrated with deep neural network (DNN) accelerator to achieve complete application acceleration. This application focuses on accelerating the pre-processing involved in inference of object detection & classification networks.

## Background

Input images are preprocessed  before being fed for inference of different deep neural networks. The pre-processing steps vary from network to network. For example, for classification networks like Resnet-50 the input image is resized to 224 x 224 size and then channel-wise mean subtraction is performed before feeding the data to the DNN accelerator. For detection networks like YOLO v3 the input image is resized to 256 x 512 size using letterbox before feeding the data to the DNN accelerator. 


[Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) provides functions optimized for FPGA devices that are drop-in replacements for standard OpenCV library functions. This application demonstrates how Vitis Vision library functions can be used to accelerate pre-processing.

## Resnet50

Currently, applications accelerating pre-processing for classification networks (Resnet-50) is provided and  can only run on ZCU102 board (device part  xczu9eg-ffvb1156-2-e) and Alveo U50 card (device part xcu50-fsvh2104-2-e). In this application, software JPEG decoder is used for loading input image. JPEG decoder transfer input image data to pre-processing kernel and the pre-processed data is transferred to the ML accelerator. Below image shows the inference pipeline.


<div align="center">
  <img width="75%" height="75%" src="./doc_images/block_dia_classification.PNG">
</div>

## ADAS detection

ADAS (Advanced Driver Assistance Systems) application
using YOLO-v3 network model is an example for object detection.
Accelerating pre-processing for YOLO-v3 is provided and can only run on ZCU102 board (device part xczu9eg-ffvb1156-2-e) and Alveo U50 card (device part xcu50-fsvh2104-2-e). In this application, software JPEG decoder is used for loading input image. Three processes are created one for image loading and running pre-processing kernel ,one for running the ML accelerator and one for generating output image. JPEG decoder transfer input image data to pre-processing kernel and the pre-processed data is transferred to the ML accelerator over a queue. ML accelerator output will be transferd over queue to create output image. Below image shows the inference pipeline.

<div align="center">
  <img width="75%" height="75%" src="./doc_images/block_dia_adasdetection.PNG">
</div>


## Running the Application on ZCU102
### Setting Up the Target
**To improve the user experience, the Vitis AI Runtime packages have been built into the board image. Therefore, user does not need to install Vitis AI
Runtime packages on the board separately.**

1. Installing a Board Image.
	* Download the SD card system image files from the following links:  
	
		[ZCU102](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2020.1-v1.3.0.img.gz) 

		For bash, same sd_card system image can be obtained from here

		```
		#XCD server
		/group/dphi_software/vitis_ai_library/r1.3/xilinx-zcu102-dpu-v2020.2-v1.3.0.img.gz
		``` 
	    
      	Note: The version of the board image should be 2020.2 or above.
	* Use Etcher software to burn the image file onto the SD card.
	* Insert the SD card with the image into the destination board.
	* Plug in the power and boot the board using the serial port to operate on the system.
	* Set up the IP information of the board using the serial port.
	You can now operate on the board using SSH.
	
2. Update the system image files.
	* Download the [waa_system_v1.3.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=waa_system_v1.3.0.tar.gz).	

		For bash, same tar file can be obtained from `/wrk/acceleration/users/maheshm/vai_1_3/waa_system_v1.3.0.tar.gz`
	* Copy the `waa_system_v1.3.0.tar.gz` to the board using scp.
	```
	scp waa_system_v1.3.0.tar.gz root@IP_OF_BOARD:~/
	```
	* Update the system image files on the target side

	**For Resnet50:**

	```
	cd ~
	tar -xzvf waa_system_v1.3.0.tar.gz
	cp waa_system_v1.3.0/sd_card_resnet50/* /mnt/sd-mmcblk0p1/
	cp /mnt/sd-mmcblk0p1/dpu.xclbin /usr/lib/
	ln -s /usr/lib/dpu.xclbin /mnt/dpu.xclbin
	cp waa_system_v1.3.0/lib/* /usr/lib/
	reboot
	```
	**For Adas Detection:**

	```
	cd ~
	tar -xzvf waa_system_v1.3.0.tar.gz
	cp waa_system_v1.3.0/sd_card_adasdetection/* /mnt/sd-mmcblk0p1/
	cp /mnt/sd-mmcblk0p1/dpu.xclbin /usr/lib/
	ln -s /usr/lib/dpu.xclbin /mnt/dpu.xclbin
	reboot
	```
	**Note that `waa_system_v1.3.0.tar.gz` can only be used for ZCU102.**


## Running the Application on Alveo U50
### Setting Up the Target

* Follow the steps mentioned [here](../../setup/alveo/u50_u50lv_u280/README.md) to setup the target. 

* Download [waa_system_u50_v1.3.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=waa_system_u50_v1.3.0.tar.gz) and update the xclbin file.

* For bash tar file can be obtained from `/wrk/acceleration/users/nkpavan/vai_1_3/vai_1_3_u50_package/waa_system_u50_v1.3.0.tar.gz`

```
	tar -xzvf waa_system_u50_v1.3.0.tar.gz
	sudo cp waa_system_u50_v1.3.0/* /usr/lib/.
```	
	
### Running The Examples
Before running the examples the model files need to be downloaded.

wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102-zcu104-r1.3.0.tar.gz -O resnet50-zcu102-zcu104-r1.3.0.tar.gz

* To download and install `resnet50` model:
	```
	  cd ${VAI_ALVEO_ROOT}/..
	  wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-u50-r1.3.0.tar.gz -O resnet50-u50-r1.3.0.tar.gz
	  For bash tar file can be obtained from `/wrk/acceleration/users/nkpavan/vai_1_3/vai_1_3_u50_package/resnet50-u50-r1.3.0.tar.gz`

	```	
	* Install the model package.
	```
	  tar -xzvf resnet50-u50-r1.3.0.tar.gz
	  sudo cp resnet50 /usr/share/vitis_ai_library/models -r
	```

* To download and install `adas detection` model:
	```
	  cd ${VAI_ALVEO_ROOT}/..
	  wget https://www.xilinx.com/bin/public/openDownload?filename=yolov3_adas_pruned_0_9-u50-r1.3.0.tar.gz -O yolov3_adas_pruned_0_9-u50-r1.3.0.tar.gz
	  For bash tar file can be obtained from `/wrk/acceleration/users/nkpavan/vai_1_3/vai_1_3_u50_package/yolov3_adas_pruned_0_9-u50-r1.3.0.tar.gz`
	```	
	* Install the model package.
	```
	  tar -xzvf yolov3_adas_pruned_0_9-u50-r1.3.0.tar.gz
	  sudo cp yolov3_adas_pruned_0_9 /usr/share/vitis_ai_library/models -r
	```
	
For resnet50_mt_py_waa example, download the images at http://image-net.org/download-images and copy 1000 images to `Vitis-AI/demo/Whole-App-Acceleration/resnet50_mt_py_waa/images` 

For bash test images can be obtained from `/wrk/acceleration/users/nkpavan/vai_1_3/vai_1_3_u50_package/resnet_test_image.jpg`

For adas_detection_waa example, download the images at https://cocodataset.org/#download and copy the images to `Vitis-AI/demo/Whole-App-Acceleration/adas_detection_waa/data`

For bash test images can be obtained from `/wrk/acceleration/users/nkpavan/vai_1_3/vai_1_3_u50_package/adas_detection_input.jpg`

3. Copy WAA examples to the board (only for ZCU102 run)

```
	scp -r Vitis-AI/demo/Whole-App-Acceleration/resnet50_mt_py_waa root@IP_OF_BOARD:~/
	scp -r Vitis-AI/demo/Whole-App-Acceleration/adas_detection_waa root@IP_OF_BOARD:~/
```

4. Compile and run the program on the target

For resnet50_mt_py_waa example, please refer to [resnet50_mt_py_waa readme](./resnet50_mt_py_waa/README.md) 

For adas_detection_waa example, please refer to [adas_detection_waa readme](./adas_detection_waa/README.md) 

### Performance:
Below table shows the comparison of througput achieved by acclerating the pre-processing pipeline on FPGA. 
For `Resnet-50`, the performance numbers are achieved by running 1K images randomly picked from ImageNet dataset.
For `Adas Detection`, the performance numbers are achieved by running 1K images randomly picked from COCO dataset. 

FPGA: ZCU102


<table style="undefined;table-layout: fixed; width: 534px">
<colgroup>
<col style="width: 119px">
<col style="width: 136px">
<col style="width: 145px">
<col style="width: 134px">
</colgroup>
  <tr>
    <th rowspan="2">Network</th>
    <th colspan="2">E2E Throughput (fps)</th>
    <th rowspan="2"><span style="font-weight:bold">Percentage improvement in throughput</span></th>
  </tr>
  <tr>
    <td>with software Pre-processing</td>
    <td>with hardware Pre-processing</td>
  </tr>

  <tr>
    <td>Resnet-50</td>
    <td>47.6</td>
    <td>64</td>
    <td>34.4%</td>
  </tr>
  
  <tr>
   <td>Adas Detection</td>
    <td>10.1</td>
    <td>15.1</td>
        <td>49.5%</td>
  </tr>
</table>

**Note that Performance numbers are computed using end-to-end latency and it depends on input image resolution. So performance numbers can vary with different images**