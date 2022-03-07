# PSMNet
Supported devices
- VCK190

## Table of Contents

- [Introduction](#Introduction)
- [Flash SD card](#Flash-SD-card)
  - [Using pre-built xclbin](#Using-pre-built-xclbin)
  - [Build flow](#Build-flow)
- [Setup](#Setup)
- [Running on VCK190](#Running-on-VCK190)

## Introduction
PSMNet application accelerating cost volume creation is provided and can only run on VCK190 platform.

## Flash SD card
PSMNet example runs in 2 different ways:
1. Using pre-built xclbin
1. Build flow

### **Using pre-built xclbin**

* Download the VCK190 SD card image file using the below link.

    [VCK190](https://www.xilinx.com/bin/public/openDownload?filename=waa_vck190_costV21_2_sd_card_vai2.0.tar.gz)

* Unzip the file and flash the .img file to SD card using tools like Etcher.

### **Build flow**

* Follow [build_flow/DPUCVDX8G_vck190](./build_flow/DPUCVDX8G_vck190/README.md)

## **Setup**

* Download the xmodel files
  ```sh
  cd ${VAI_HOME}/Whole-App-Acceleration/apps/psmnet

  wget https://www.xilinx.com/bin/public/openDownload?filename=PSMNet_pruned_0_pt-vck190-r2.0.0.tar.gz -O PSMNet_pruned_0_pt-vck190-r2.0.0.tar.gz
  tar -xzvf PSMNet_pruned_0_pt-vck190-r2.0.0.tar.gz && rm PSMNet_pruned_0_pt-vck190-r2.0.0.tar.gz

  wget https://www.xilinx.com/bin/public/openDownload?filename=PSMNet_pruned_1_pt-vck190-r2.0.0.tar.gz -O PSMNet_pruned_1_pt-vck190-r2.0.0.tar.gz
  tar -xvzf PSMNet_pruned_1_pt-vck190-r2.0.0.tar.gz && rm PSMNet_pruned_1_pt-vck190-r2.0.0.tar.gz

  wget https://www.xilinx.com/bin/public/openDownload?filename=PSMNet_pruned_2_pt-vck190-r2.0.0.tar.gz -O PSMNet_pruned_2_pt-vck190-r2.0.0.tar.gz
  tar -xvzf PSMNet_pruned_2_pt-vck190-r2.0.0.tar.gz && rm PSMNet_pruned_2_pt-vck190-r2.0.0.tar.gz
  ```

  * Download the images at [vitis_ai_library_r2.0.x_images.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r2.0.0_images.tar.gz)
  ```sh
  cd ${VAI_HOME}/Whole-App-Acceleration/apps/psmnet
  wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r2.0.0_images.tar.gz -O vitis_ai_library_r2.0.0_images.tar.gz
  tar -xzvf vitis_ai_library_r2.0.0_images.tar.gz && rm vitis_ai_library_r2.0.0_images.tar.gz
  cp -r samples/dpu_task/psmnet/* .
  rm -rf samples
  ```

  * Copy ${VAI_HOME}/Whole-App-Acceleration/apps/psmnet directory to the BOOT partition of the SD Card.

  * Insert SD_CARD on the VCK190 board. After the linux boot, run:

  * compile `psmnet` example.
    ```sh
    cd /media/sd-mmcblk0p1/
    bash -x build.sh
    ```

  * If the compilation process does not report any error and the executable files `./demo_psmnet` and `./test_performance_psmnet` are generated.

## *Running on **VCK190***

#### Run on VCK190:
```sh
export XLNX_ENABLE_FINGERPRINT_CHECK=0
env ./demo_psmnet demo_psmnet_left.png demo_psmnet_right.png
```
> After execution, the result is stored as an image: result_psmnet_0.jpg

#### Run for the performance
```sh
./test_performance_psmnet -t 3 -s 60 test_performance_psmnet.list
```
