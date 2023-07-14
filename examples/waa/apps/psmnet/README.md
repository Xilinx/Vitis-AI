# PSMNet
Supported devices
- VCK190

:pushpin: **Note:** Use VAI2.5 setup to run this applicaion

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

### **Build flow**

* Follow [build_flow/DPUCVDX8G_vck190](./build_flow/DPUCVDX8G_vck190/README.md)

### **Using pre-built xclbin**

* Installing a Board Image

  * Download the VCK190 SD card image file using the below link.

    [VCK190](https://www.xilinx.com/bin/public/openDownload?filename=waa_vck190_psmnet_2_5.img.gz)

  * Unzip the file and flash the .img file to SD card using tools like Etcher.

* Installing Vitis AI Runtime on the Evaluation Board

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

* Cross-compile Psmnet example

  * Download the [sdk-2022.1.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2022.1.0.0.sh)

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

  * Cross compile `psmnet` example
    ```sh
    bash -x build.sh
    ```
    If the compilation process does not report any error and the executable file `demo_psmnet` and `test_performance_psmnet` is generated , then the host environment is installed correctly.   

* Download the xmodel files
  ```sh
  cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/psmnet

  wget https://www.xilinx.com/bin/public/openDownload?filename=PSMNet_pruned_0_pt-vck190-r2.5.0.tar.gz -O PSMNet_pruned_0_pt-vck190-r2.5.0.tar.gz
  tar -xzvf PSMNet_pruned_0_pt-vck190-r2.5.0.tar.gz && rm PSMNet_pruned_0_pt-vck190-r2.5.0.tar.gz

  wget https://www.xilinx.com/bin/public/openDownload?filename=PSMNet_pruned_1_pt-vck190-r2.5.0.tar.gz -O PSMNet_pruned_1_pt-vck190-r2.5.0.tar.gz
  tar -xvzf PSMNet_pruned_1_pt-vck190-r2.5.0.tar.gz && rm PSMNet_pruned_1_pt-vck190-r2.5.0.tar.gz

  wget https://www.xilinx.com/bin/public/openDownload?filename=PSMNet_pruned_2_pt-vck190-r2.5.0.tar.gz -O PSMNet_pruned_2_pt-vck190-r2.5.0.tar.gz
  tar -xvzf PSMNet_pruned_2_pt-vck190-r2.5.0.tar.gz && rm PSMNet_pruned_2_pt-vck190-r2.5.0.tar.gz
  ```

* Download the images at [vitis_ai_library_r2.5.0_images.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r2.5.0_images.tar.gz)
  ```sh
  cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/psmnet
  wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r2.5.0_images.tar.gz -O vitis_ai_library_r2.5.0_images.tar.gz
  tar -xzvf vitis_ai_library_r2.5.0_images.tar.gz && rm vitis_ai_library_r2.5.0_images.tar.gz
  cp -r samples/dpu_task/psmnet/* .
  rm -rf samples
  ```
* Run **psmnet** example

  * Copy the following contents of ${VAI_HOME}/examples/Whole-App-Acceleration/apps/psmnet directory to the BOOT partition `/run/media/mmcblk0p1/` of the SD Card

    ```sh
    demo_psmnet
    images
    demo_psmnet_left.png
    demo_psmnet_right.png
    PSMNet_pruned_0_pt
    PSMNet_pruned_1_pt
    PSMNet_pruned_2_pt
    test_performance_psmnet
    test_performance_psmnet.list 
    ```

  * Insert SD_CARD on the VCK190 board. After the linux boot, run:

  * For psmnet demo
    ```sh
    export XLNX_ENABLE_FINGERPRINT_CHECK=0
    env ./demo_psmnet demo_psmnet_left.png demo_psmnet_right.png
    ```
    > After execution, the result is stored as an image: result_psmnet_0.jpg

  * For performance test
    ```sh
    ./test_performance_psmnet -t 3 -s 60 test_performance_psmnet.list
    ```
