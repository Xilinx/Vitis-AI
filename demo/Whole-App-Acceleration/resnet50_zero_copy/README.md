## Resnet-50
:pushpin: **Note:** This application can be run only on ZCU102.

## Table of Contents

- [Introduction](#Introduction)
- [Setting Up and Running on ZCU102](#Setting-Up-and-Running-on-ZCU102)
    - [Setting Up the Target](#Setting-Up-the-Target-ZCU102)
    - [Running the application](#Running-the-application-on-ZCU102-board)
- [Performance](#Performance)    

## Introduction
Currently, applications accelerating pre-processing for classification networks (Resnet-50) is provided and  can only run on ZCU102 board (device part  xczu9eg-ffvb1156-2-e). In this application, software JPEG decoder is used for loading input image. JPEG decoder transfer input image data to pre-processing kernel and the pre-processed data is directly stored at ML accelerator physical address. Below image shows the inference pipeline.

:pushpin: **Note:** In this application pre-processed data is directly stored at ML accelerator physical address. Hence avoiding device to host data transfers.


<div align="center">
  <img width="75%" height="75%" src="../doc_images/block_dia_classification.PNG">
</div>

## Setting Up and Running on ZCU102

### Setting Up the Target ZCU102

#### Installing board image 
* Download the SD card system image files from the following links: 
      [ZCU102](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-waa-classification-v2020.2-v1.4.0.img)

Note: The version of the board image should be 2020.2 or above.

* Use Etcher software to burn the image file onto the SD card.
#### Installing Vitis AI Runtime on the Evaluation Board
* Download the  [Vitis AI Runtime 1.4.0](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.4.0.tar.gz). 
* Untar the runtime packet and copy the following folder to the board using scp.
```
	tar -xzvf vitis-ai-runtime-1.4.0.tar.gz
	scp -r vitis-ai-runtime-1.4.0/2020.2/aarch64/centos/ root@IP_OF_BOARD:~/
```
* Install the Vitis AI Runtime on the evaluation board. Execute the following command.
```
	cd ~/centos
	bash setup.sh
```
#### (Optional) Cross-compile WAA example
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

* Download the [vitis_ai_2020.2-r1.4.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2020.2-r1.4.0.tar.gz) and install it to the petalinux system.
    ```
    tar -xzvf vitis_ai_2020.2-r1.4.0.tar.gz -C ~/petalinux_sdk/sysroots/aarch64-xilinx-linux
    ```

* Cross compile `resnet50_zero_copy` example.
    ```
    cd  Vitis-AI/demo/Whole-App-Acceleration/resnet50_zero_copy
    bash -x build.sh
    ```
    If the compilation process does not report any error and the executable file `resnet50_zero_copy` is generated , then the host environment is installed correctly.

### Running the application on ZCU102 board

#### Download Model files for Resnet50
```
%	cd Vitis-AI/demo/Whole-App-Acceleration/resnet50_zero_copy
%	mkdir model_zcu102
%	cd model_zcu102
%	wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102_zcu104_kv260-r1.4.0.tar.gz -O resnet50-zcu102_zcu104_kv260-r1.4.0.tar.gz
%	tar -xzvf resnet50-zcu102_zcu104_kv260-r1.4.0.tar.gz
```

#### Run example
* Download test images
    Download the images at http://image-net.org/download-images and copy 1000 images to `Vitis-AI/demo/Whole-App-Acceleration/resnet50_zero_copy/img`

* Copy the directory `Vitis-AI/demo/Whole-App-Acceleration/resnet50_zero_copy` to the BOOT partition of the SD Card.
    ```
	  scp -r Vitis-AI/demo/Whole-App-Acceleration/resnet50_zero_copy root@IP_OF_BOARD:~/
    ```

* Please insert SD_CARD on the ZCU102 board. After the linux boot, run:
```
cd ~/resnet50_zero_copy
cp /media/sd-mmcblk0p1/dpu.xclbin /usr/lib/
export XILINX_XRT=/usr
echo 1 > /proc/sys/kernel/printk
```
* Run resnet50 without waa
  ```
  ./resnet50_zero_copy model_zcu102/resnet50/resnet50.xmodel 0 0
  ```
* Run resnet50 with waa
  ```
  ./resnet50_zero_copy model_zcu102/resnet50/resnet50.xmodel 1 0
  ```  
* Run for profiling
  ```
  ./resnet50_zero_copy model_zcu102/resnet50/resnet50.xmodel <0 for sw / 1 for hw pre> 1
  ```  

### Performance
Below table shows the comparison of throughput (without software JPEG decoder) achieved by accelerating the pre-processing pipeline on FPGA.
For `Resnet-50`, the performance numbers are achieved by running 1K images randomly picked from ImageNet dataset.

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
    <td>55.7</td>
    <td>72.29</td>
    <td>29.7%</td>
  </tr>


</table>

**Note that Performance numbers are computed using end-to-end latency and it depends on input image resolution. So performance numbers can vary with different images**  
