# Segmentation Network Demo

:pushpin: **Note:** This application can be run only on VCK190 platform.

## Table of Contents

- [Introduction](#introduction)
- [Running the Application](#running-the-application)


## Introduction
This application demonstrates the acceleration of pre-processing of image segmentation application.

## Running the Application
* Download the VCK190 SD card image file using the below link.

	[VCK190](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-vck190-dpu-v2020.2-v1.4.0.img.gz)
 
  Please note that Xilinx account sign-in is required to download the above file.
* Unzip the file and flash the .img file to SD card using tools like Etcher.

* Download the WAA package 

```
wget https://www.xilinx.com/bin/public/openDownload?filename=waa_versal_sd_card_vai2.0.tar.gz -O waa_versal_sd_card_vai2.0.tar.gz

tar -xzvf waa_versal_sd_card_vai2.0.tar.gz
```
* copy the contents of the WAA package to the SD card. The xmodel file is also present in the package.

* Copy content of  `${VAI_HOME}/Whole-App-Acceleration/apps/segmentation` directory to the BOOT partition of the SD Card.

* Please insert SD_CARD on the vck190 board. After the linux boot, run:

  * compile `segmentation` example.
    ```
    cd /media/sd-mmcblk0p1/
    bash -x board_build.sh
    ```

* Run the below command

    ```
    ./segmentation.exe fcn8_vck190.xmodel
    ```





