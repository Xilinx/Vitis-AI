# Segmentation Network Demo

:pushpin: **Note:** This application can be run only on VCK190 platform.

:pushpin: **Note:** Use VAI2.5 setup to run this applicaion


## Table of Contents

- [Introduction](#Introduction)
- [Setting SD Card](#Setting-SD-Card)
- [Install Vitis AI Runtime on the Evaluation Board](#Install-Vitis-AI-Runtime-on-the-Evaluation-Board)
- [Cross Compile Segmentation example](#Cross-Compile-Segmentation-example)
- [Running the Application](#Running-the-Application)


## Introduction
This application demonstrates the acceleration of pre-process and post-process of image segmentation application.

## Setting SD Card 

* The xclbin including pre-process and post-process accelerator and DPU is already built and packaged to create a sd_card.img file. Download the VCK190 SD card image file using the below link.

	[VCK190](https://www.xilinx.com/bin/public/openDownload?filename=segmentation_sd_card.tar.gz)
 
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

## Running the Application

* Copy the following contents of  `${VAI_HOME}/examples/Whole-App-Acceleration/apps/segmentation` directory to the BOOT partition of the SD Card.

  ```sh
  executable file
  xmodel file
  images
  ```

* Please insert SD_CARD on the vck190 board. After the linux boot, run:

    ```sh
  ./segmentation <xmodel_path> <test_image_path> <number of threads (from 1 to 6)> <post_process_type(0:hw, 1:cpu)> <preprocess_type(0:hw, 1:cpu)>
    ```
## Performance

Below table shows the comparison of performance achieved by accelerating the pre-process and post-process pipeline on FPGA.
The performance numbers are achieved by running the application over 500 FHD images.

<table style="undefined;table-layout: fixed; width: 664px">
<colgroup>
<col style="width: 119px">
<col style="width: 136px">
<col style="width: 145px">
<col style="width: 134px">
</colgroup>
  <tr>
    <th rowspan="2">Threads</th>
    <th colspan="2">Performance (FPS)</th>
    <th rowspan="2"><span style="font-weight:bold">Improvement</span></th>
  </tr>
  <tr>
    <td>with software pre and post process</td>
    <td>with hardware pre and post process</td>
  </tr>

  <tr>
    <td>1</td>
    <td>2.6</td>
    <td>5.59</td>
    <td>115 %</td>
  </tr>
    <td>2</td>
    <td>4.92</td>
    <td>6.6</td>
    <td>34.14 %</td>
  <tr>
  <td>3</td>
    <td>5.0</td>
    <td>6.7</td>
    <td>34 %</td>
  </tr> 

</table>

:pushpin: **Note:** The above performance numbers doesn't consider the image read time and NPC1 is used in pre-process accelerator.


:pushpin: **Note:** The performance numbers are computed using end-to-end latency and it depends on input image resolution. So performance numbers can vary with different images.
