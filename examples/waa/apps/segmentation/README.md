# Segmentation Network Demo

:pushpin: **Note:** This application can be run only on VCK190 platform.

## Table of Contents

- [Introduction](#Introduction)
- [Setting SD Card](#Setting-SD-Card)
- [Install Vitis AI Runtime on the Evaluation Board](#Install-Vitis-AI-Runtime-on-the-Evaluation-Board)
- [Cross Compile Segmentation example](#Cross-Compile-Segmentation-example)
- [Running the Application](#Running-the-Application)


## Introduction
This application demonstrates the acceleration of pre-processing of image segmentation application.

## Setting SD Card 

* The xclbin including pre-processing accelerator and DPU is already built and packaged to create a sd_card.img file. Download the VCK190 SD card image file using the below link.

	[VCK190](https://www.xilinx.com/bin/public/openDownload?filename=segmentation_sd_card_2_5.tar.gz)
 
  Please note that Xilinx account sign-in is required to download the above file.
* Unzip the file and flash the .img file to SD card using tools like Etcher.

## Install Vitis AI Runtime on the Evaluation Board

  * Download the [Vitis AI Runtime 2.5.0](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-2.5.0.tar.gz)

  * Untar the runtime packet and copy the following folder to the board using scp

    ```sh
    tar -xzvf vitis-ai-runtime-2.5.0.tar.gz
    scp -r vitis-ai-runtime-2.5.0/2022.1/aarch64/centos root@IP_OF_BOARD:~/ 
    ```

   * Install the Vitis AI Runtime on the evaluation board. Execute the following command

      ```sh
        cd ~/centos
        bash setup.sh
      ```

## Cross Compile Segmentation example

  Download the [sdk-2022.1.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2022.1.0.0.sh)

  * Install the cross-compilation system environment, follow the prompts to install

    **Please install it on your local host linux system, not in the docker system**

    ```sh
    ./sdk-2022.1.0.0.sh
    ```
    Note that the `~/petalinux_sdk` is recommended for the installation. Regardless of the path you choose for the installation, make sure the path has read-write permissions. Here we install it under `~/petalinux_sdk`.

  * Download the [vitis_ai_2022.1-r2.5.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2022.1-r2.5.0.tar.gz) and install it to the petalinux system 
    ```sh
    tar -xzvf vitis_ai_2022.1-r2.5.0.tar.gz -C ~/petalinux_sdk/sysroots/cortexa72-cortexa53-xilinx-linux
    ```   

  * When the installation is complete, follow the prompts and execute the following command

    ```sh
    . ~/petalinux_sdk/environment-setup-cortexa72-cortexa53-xilinx-linux
    ```
    Note that if you close the current terminal, you need to re-execute the above instructions in the new terminal interface.

  * Cross compile `Segmentation` example
    ```sh
    ./build.sh
    ```
    If the compilation process does not report any error and the executable file `segmentation` is generated , then the host environment is installed correctly.   


## Download the WAA package

* The xmodel file is present in this package.

  ```sh
  wget https://www.xilinx.com/bin/public/openDownload?filename=waa_versal_sd_card_vai2.5.tar.gz -O waa_versal_sd_card_vai2.5.tar.gz

  tar -xzvf waa_versal_sd_card_vai2.5.tar.gz
  ```
* Copy the content of the WAA package to the SD card.

## Download images

* Download the images at https://cocodataset.org/#download or any other repositories and copy the images to ${VAI_HOME}/examples/Whole-App-Acceleration/apps/segmentation/images directory. In the following performance test we used COCO dataset.


## Running the Application

* Copy the following contents of  `${VAI_HOME}/examples/Whole-App-Acceleration/apps/segmentation` directory to the BOOT partition of the SD Card.

  ```sh
  executable file
  xmodel file
  images
  ```

* Please insert SD_CARD on the vck190 board. After the linux boot, run:

    ```sh
  ./segmentation <xmodel_path> <test_image_path> <number of threads (from 1 to 6)> <use_post_proc(1:yes, 0:no)> <preprocess_type(0:hw, 1:cpu)>
    ```
## Performance

Below table shows the comparison of performance achieved by accelerating the pre-processing pipeline on FPGA.
The performance numbers are achieved by running the application over 1000 images. Preprocess and dpu execution time has been considered for throughput calculation. 

<table style="undefined;table-layout: fixed; width: 664px">
<colgroup>
<col style="width: 119px">
<col style="width: 136px">
<col style="width: 145px">
<col style="width: 134px">
</colgroup>
  <tr>
    <th rowspan="2">Device</th>
    <th colspan="2">Performance (FPS)</th>
    <th rowspan="2"><span style="font-weight:bold">Improvement</span></th>
  </tr>
  <tr>
    <td>with software Pre-processing</td>
    <td>with hardware Pre-processing</td>
  </tr>

  <tr>
    <td>VCK190</td>
    <td>25.10</td>
    <td>50.67</td>
    <td>101.87 %</td>
  </tr>




