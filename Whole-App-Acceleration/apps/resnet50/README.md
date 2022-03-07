# Resnet-50
Supported Devices
- ZCU102
- ALVEO-U50
- ALVEO-U200
- ALVEO-U280
- VCK190

## Table of Contents

- [Introduction](#Introduction)
- [Run Resnet50 example](#Run-Resnet50-example)
  - [Using pre-built xclbin](#Using-pre-built-xclbin)
    - [Setting up and running on ZCU102](#Setting-up-and-running-on-ZCU102)
    - [Setting up and running on U50 U200 U280](#Setting-up-and-running-on-U50-U200-U280)
    - [Setting up and running on VCK190](#Setting-up-and-running-on-VCK190)        
  - [Build flow](#Build-flow)
- [Performance](#Performance)    

## Introduction
Currently, applications accelerating pre-processing for classification networks (Resnet-50) is provided and can only run on ZCU102/U50/U280/U200/VCK190 platforms. In this application, software JPEG decoder is used for loading input image. JPEG decoder transfer input image data to pre-processing kernel and the pre-processed data is directly stored at the ML accelerator input physical address. Below image shows the inference pipeline.

:pushpin: **Note:** In this application pre-processed data is directly stored at ML accelerator physical address. Hence avoiding device to host data transfers.

<div align="center">
  <img width="75%" height="75%" src="./block_dia_classification.PNG">
</div>


## Run Resnet50 example
---
Resnet50 example runs with 2 different ways. 
1. Using pre-built xlcbin 
1. Build flow

### **Using pre-built xclbin**

#### *Setting up and running on **ZCU102***
* Installing a Board Image.
	* Download the SD card system image files from the following links:  
	
      [ZCU102](https://www.xilinx.com/bin/public/openDownload?filename=waa_zcu102_resnet50_v2_0_0.img.gz)   
	    
      :pushpin: **Note:** The version of the board image should be 2021.2 or above.
	* Use Etcher software to burn the image file onto the SD card.
	* Insert the SD card with the image into the destination board.
	* Plug in the power and boot the board using the serial port to operate on the system.
	* Set up the IP information of the board using the serial port.
	You can now operate on the board using SSH.
  
* Installing Vitis AI Runtime on the Evaluation Board

  * Download the [Vitis AI Runtime 2.0.0](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-2.0.0.tar.gz). 

  * Untar the runtime packet and copy the following folder to the board using scp.
    ```sh
	  tar -xzvf vitis-ai-runtime-2.0.0.tar.gz
	  scp -r vitis-ai-runtime-2.0.0/2021.2/aarch64/centos root@IP_OF_BOARD:~/
    ```
  * Install the Vitis AI Runtime on the evaluation board. Execute the following command.
    ```sh
	  cd ~/centos
	  bash setup.sh
    ```
* Cross-compile Resnet50 example
  * Download the [sdk-2021.2.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2021.2.0.0.sh)

  * Install the cross-compilation system environment, follow the prompts to install. 

    **Please install it on your local host linux system, not in the docker system.**
    ```sh
    ./sdk-2021.2.0.0.sh
    ```
    Note that the `~/petalinux_sdk` path is recommended for the installation. Regardless of the path you choose for the installation, make sure the path has read-write permissions. 
    Here we install it under `~/petalinux_sdk`.

  * When the installation is complete, follow the prompts and execute the following command.
    ```sh
    . ~/petalinux_sdk/environment-setup-cortexa72-cortexa53-xilinx-linux
    ```
    Note that if you close the current terminal, you need to re-execute the above instructions in the new terminal interface.

  * Download the [vitis_ai_2021.2-r2.0.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2021.2-r2.0.0.tar.gz) and install it to the petalinux system.
    ```sh
    tar -xzvf vitis_ai_2021.2-r2.0.0.tar.gz -C ~/petalinux_sdk/sysroots/cortexa72-cortexa53-xilinx-linux
    ```

  * Cross compile `resnet50` example.
    ```sh
    cd  ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50
    bash -x app_build.sh
    ```
      If the compilation process does not report any error and the executable file `./bin/resnet50.exe` is generated , then the host environment is installed correctly.

* Download Model files for Resnet50

    ```sh
    cd ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50
    mkdir model_zcu102
    cd model_zcu102
    wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102_zcu104_kv260-r2.0.0.tar.gz -O resnet50-zcu102_zcu104_kv260-r2.0.0.tar.gz
    tar -xzvf resnet50-zcu102_zcu104_kv260-r2.0.0.tar.gz
    ```

* Run Resnet50 Example

  This part is about how to run the Resnet50 example on zcu102 board.

  * Download the images at http://image-net.org/download-images and copy images to ` ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/img` directory 

  * Copy following contents of  ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50 directory to the BOOT partition of the SD Card.
    ```sh
        bin
        model_zcu102
        img
        app_test.sh
        words.txt
    ```


  * Please insert SD_CARD on the ZCU102 board. After the linux boot, run:

  * Performance test with & without WAA

    ```sh
    cd /media/sd-mmcblk0p1/
    export XLNX_VART_FIRMWARE=/media/sd-mmcblk0p1/dpu.xclbin
    
    ./app_test.sh --xmodel_file ./model_zcu102/resnet50/resnet50.xmodel --image_dir ./img/ --performance_diff
    
    # Expect similar output
        Running Performance Diff: 

          Running Application with Software Preprocessing 

          E2E Performance: 49.70 fps
          Pre-process Latency: 7.68 ms
          Execution Latency: 11.72 ms
          Post-process Latency: 0.70 ms

          Running Application with Hardware Preprocessing 

          E2E Performance: 74.53 fps
          Pre-process Latency: 1.04 ms
          Execution Latency: 11.66 ms
          Post-process Latency: 0.69 ms

          The percentage improvement in throughput is 49.95 %
       
    ```

  * Functionality test with single image using WAA
    ```sh
    ./app_test.sh --xmodel_file ./model_zcu102/resnet50/resnet50.xmodel --image_dir ./img/ --verbose

    # Expect similar output:
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I0712 10:16:33.656128  1587 main.cc:465] create running for subgraph: subgraph_conv1
    Number of images in the image directory is: 1
    top[0] prob = 0.829972  name = sea snake
    top[1] prob = 0.068128  name = hognose snake, puff adder, sand viper
    top[2] prob = 0.032181  name = water snake
    top[3] prob = 0.015201  name = horned viper, cerastes, sand viper, horned asp, Cerastes cornutus
    top[4] prob = 0.015201  name = American alligator, Alligator mississipiensis
    ```

  * Functionality test with single image without WAA (software preprocessing)
    ```sh
    ./app_test.sh --xmodel_file ./model_zcu102/resnet50/resnet50.xmodel --image_dir ./img/ --verbose --use_sw_pre_proc

    # Expect similar output:
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I0712 10:16:42.329468  1612 main.cc:465] create running for subgraph: subgraph_conv1
    Number of images in the image directory is: 1
    top[0] prob = 0.808481  name = sea snake
    top[1] prob = 0.066364  name = hognose snake, puff adder, sand viper
    top[2] prob = 0.031348  name = water snake
    top[3] prob = 0.031348  name = American alligator, Alligator mississipiensis
    top[4] prob = 0.024414  name = African crocodile, Nile crocodile, Crocodylus niloticus
    ```

#### *Setting up and running on **U50, U200, U280***

:pushpin: **Note:** Refer to [Setup Alveo Accelerator Card](../../../setup/alveo) to set up the Alveo Card.

:pushpin: **Note:** The docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../README.md#Installation)

* Download and install xclbin.
  * To install **U50** xclbin, download the [waa_u50_xclbins_v2_0_0](https://www.xilinx.com/bin/public/openDownload?filename=waa_u50_xclbins_v2_0_0.tar.gz) xclbin tar and install xclbin.
	```sh
	sudo tar -xzvf waa_u50_xclbins_v2_0_0.tar.gz --directory /
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/waa_u50_xclbins_v2_0_0/resnet50/dpu.xclbin
	```
  * To install **U200** xclbin, download the [waa_u200_xclbins_v2_0_0](https://www.xilinx.com/bin/public/openDownload?filename=waa_u200_xclbins_v2_0_0.tar.gz) xclbin tar and install xclbin.
	```sh
	sudo tar -xzvf waa_u200_xclbins_v2_0_0.tar.gz --directory /
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/waa_u200_xclbins_v2_0_0/resnet50/dpu.xclbin
	```
  * To install **U280** xclbin, download the [waa_u280_xclbins_v2_0_0](https://www.xilinx.com/bin/public/openDownload?filename=waa_u280_xclbins_v2_0_0.tar.gz) xclbin tar and install xclbin.
	```sh
	sudo tar -xzvf waa_u280_xclbins_v2_0_0.tar.gz --directory /
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/waa_u280_xclbins_v2_0_0/resnet50/dpu.xclbin
	```
* Download and install resnet50 model.
  * To install model file for **U50 & U280**
	  ```sh
    mkdir -p ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/model_dir
    wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-u50-u50lv-u280-DPUCAHX8H-r1.4.1.tar.gz -O resnet50-u50-u50lv-u280-DPUCAHX8H-r1.4.1.tar.gz
    tar -xzvf resnet50-u50-u50lv-u280-DPUCAHX8H-r1.4.1.tar.gz -C ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/model_dir
	```
  * To install model file for **U200**
	  ```sh
    mkdir -p ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/model_dir
	wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-u200-u250-r1.4.0.tar.gz -O resnet50-u200-u250-r1.4.0.tar.gz
    tar -xzvf resnet50-u200-u250-r1.4.0.tar.gz -C ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/model_dir
	  ```
* Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012/) using [Collective Knowledge (CK)](https://github.com/ctuning).

	```sh
  # Activate conda env
  conda activate vitis-ai-caffe
  python -m ck pull repo:ck-env
  python -m ck install package:imagenet-2012-val-min

  # We don't need conda env for running examples with this DPU
  conda deactivate
	```
  :pushpin: **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

* Building Resnet50 application
	```sh
  cd ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50
  ./app_build.sh
	```

  If the compilation process does not report any error then the executable file `./bin/resnet50.exe` is generated.    

* Run Resnet50 Example
  * Performance test with & without WAA

    ```sh
    export XLNX_ENABLE_FINGERPRINT_CHECK=0
    ./app_test.sh --xmodel_file ./model_dir/resnet50/resnet50.xmodel --image_dir ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ --performance_diff

    # Expect similar output:
      Running Performance Diff: 

          Running Application with Software Preprocessing 

          E2E Performance: 167.39 fps
          Pre-process Latency: 2.78 ms
          Execution Latency: 2.84 ms
          Post-process Latency: 0.35 ms

          Running Application with Hardware Preprocessing 

          E2E Performance: 212.95 fps
          Pre-process Latency: 1.42 ms
          Execution Latency: 2.78 ms
          Post-process Latency: 0.48 ms

          The percentage improvement in throughput is 27.22 %
    ```

  * Functionality test with single image using WAA
    ```sh
    ./app_test.sh --xmodel_file ./model_dir/resnet50/resnet50.xmodel --image_dir ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ --verbose

    # Expect similar output:
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I0712 10:16:33.656128  1587 main.cc:465] create running for subgraph: subgraph_conv1
    Number of images in the image directory is: 1
    top[0] prob = 0.829972  name = sea snake
    top[1] prob = 0.068128  name = hognose snake, puff adder, sand viper
    top[2] prob = 0.032181  name = water snake
    top[3] prob = 0.015201  name = horned viper, cerastes, sand viper, horned asp, Cerastes cornutus
    top[4] prob = 0.015201  name = American alligator, Alligator mississipiensis
    ```

  * Functionality test with single image without WAA (software preprocessing)
    ```sh
    ./app_test.sh --xmodel_file ./model_dir/resnet50/resnet50.xmodel --image_dir ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ --verbose --use_sw_pre_proc

    # Expect similar output:
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I0712 10:16:42.329468  1612 main.cc:465] create running for subgraph: subgraph_conv1
    Number of images in the image directory is: 1
    top[0] prob = 0.808481  name = sea snake
    top[1] prob = 0.066364  name = hognose snake, puff adder, sand viper
    top[2] prob = 0.031348  name = water snake
    top[3] prob = 0.031348  name = American alligator, Alligator mississipiensis
    top[4] prob = 0.024414  name = African crocodile, Nile crocodile, Crocodylus niloticus
    ```

#### *Setting up and running on VCK190*

* Download the VCK190 SD card image file using the below link.

	[VCK190](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-vck190-dpu-v2020.2-v1.4.0.img.gz)
 
  Please note that Xilinx account sign-in is required to download the above file.
* Unzip the file and flash the .img file to SD card using tools like Etcher.

* Download the WAA package 

```sh
wget https://www.xilinx.com/bin/public/openDownload?filename=waa_versal_resnet50_v2_0_0.tar.gz -O waa_versal_resnet50_v2_0_0.tar.gz

tar -xzvf waa_versal_resnet50_v2_0_0.tar.gz
```
* Copy following contents of the WAA package to the SD card to the location `/media/sd-mmcblk0p1/`.The xmodel file is also present in the package. Create `/media/sd-mmcblk0p1/resnet50` and move model_vck190 to this folder.
    ```
        BOOT.BIN
        dpu.xclbin
        include
        model_vck190
    ```

* Run Resnet50 Example

  This part is about how to run the Resnet50 example on vck190 board.

  * Download the images at http://image-net.org/download-images and copy images to ` ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/img` directory. 
  
  * Copy ` waa_versal_resnet50_v2_0_0/include` from the WAA package to ` ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/src`

  * Copy following contents of  ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50 directory to the BOOT partition `/media/sd-mmcblk0p1/resnet50` of the SD Card.
    ```
        src
        img
        app_build.sh
        app_test.sh
        words.txt
    ```


  * Please insert SD_CARD on the VCK190 board. After the linux boot, run the following.

  * Compile `resnet50` example.
    ```sh
    cd /media/sd-mmcblk0p1/
    bash -x app_build.sh
    ```
      If the compilation process does not report any error and the executable file `./bin/resnet50.exe` is generated.

  * Performance test with & without WAA
    ```sh
    cd /media/sd-mmcblk0p1/
    export XLNX_VART_FIRMWARE=/media/sd-mmcblk0p1/dpu.xclbin
    ./app_test.sh --xmodel_file ./model_vck190/resnet_v1_50.xmodel --image_dir ./img/ --performance_diff
    
    # Expect similar output
        Running Performance Diff: 

          Running Application with Software Preprocessing 

          E2E Performance: 208.768 fps
          Pre-process Latency: 3.634 ms
          Execution Latency: 0.779 ms
          Post-process Latency: 0.37 ms

          Running Application with Hardware Preprocessing 

          E2E Performance: 626.566 fps
          Pre-process Latency: 0.455 ms
          Execution Latency: 0.761 ms
          Post-process Latency: 0.371 ms

          The percentage improvement in throughput is 200.12 %
		  
    ```
  * Functionality test with single image using WAA
    ```sh
    cd /media/sd-mmcblk0p1/
    ./app_test.sh --xmodel_file ./model_vck190/resnet_v1_50.xmodel --image_dir ./img/ --verbose

    # Expect similar output:
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I0119 12:50:58.796797  1135 main.cc:510] create running for subgraph: subgraph_resnet_v1_50/block1/unit_1/bottleneck_v1/add
    Number of images in the image directory is: 1
    top[0] prob = 0.992312  name = brain coral
    top[1] prob = 0.004055  name = coral reef
    top[2] prob = 0.000905  name = puffer, pufferfish, blowfish, globefish
    top[3] prob = 0.000905  name = eel
    top[4] prob = 0.000427  name = rock beauty, Holocanthus tricolor
    ```

  * Functionality test with single image without WAA (software preprocessing)
    ```sh
    cd /media/sd-mmcblk0p1/
    ./app_test.sh --xmodel_file ./model_vck190/resnet_v1_50.xmodel --image_dir ./img/ --verbose --use_sw_pre_proc

    # Expect similar output:
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I0119 12:50:28.357049  1133 main.cc:510] create running for subgraph: subgraph_resnet_v1_50/block1/unit_1/bottleneck_v1/add
    Number of images in the image directory is: 1
    top[0] prob = 0.990261  name = brain coral
    top[1] prob = 0.005196  name = coral reef
    top[2] prob = 0.001159  name = puffer, pufferfish, blowfish, globefish
    top[3] prob = 0.000903  name = eel
    top[4] prob = 0.000427  name = rock beauty, Holocanthus tricolor
    ```

### Build flow
Both the pre-processing accelerator and DPU are built from sources.

| No. | Build flow                    | Device          | Documentation                                                                          |
|-----|-------------------------------|-----------------|----------------------------------------------------------------------------------------|
| 1   | DPUCZDX8G  | ZCU102          | [DPUCZDX8G_zcu102](./build_flow/DPUCZDX8G_zcu102/README.md)        |
| 2   | DPUCAHX8H  | Alveo-U50, U280 | [DPUCAHX8H_u50_u280](./build_flow/DPUCAHX8H_u50_u280/README.md)    |
| 3   | DPUCADF8H  | Alveo-U200      | [DPUCADF8H_u200](./build_flow/DPUCADF8H_u200/README.md)            |
| 4   | DPUCVDX8G  | VCK190      | [DPUCVDX8G_vck190](./build_flow/DPUCVDX8G_vck190/README.md)            |

## Performance
Below table shows the comparison of performance achieved by accelerating the pre-processing pipeline on FPGA.
For `Resnet-50`, the performance numbers are achieved by running 500 images randomly picked from ImageNet dataset.



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
    <td>ZCU102</td>
    <td>49.70</td>
    <td>74.53</td>
    <td>49.95 %</td>
  </tr>

  <tr>
    <td>U50</td>
    <td>146.24</td>
    <td>182.58</td>
    <td>24.85 %</td>
  </tr>

  <tr>
    <td>U200</td>
    <td>149.68</td>
    <td>187.30</td>
    <td>25.13 %</td>
  </tr>

  <tr>
    <td>U280</td>
    <td>167.39</td>
    <td>212.95</td>
    <td>27.22 %</td>
  </tr>

  <tr>
    <td>VCK190 </td>
    <td>208.768 </td>
    <td>626.566 </td>
    <td>200.12 %</td>
  </tr>

</table>

:pushpin: **Note:** The above performance numbers doesn't consider the image read time.

> System with Intel&reg; Xeon&reg; Silver 4116 CPU @ 2.10GHz is used for U200 and U280 tests while Intel&reg; Xeon&reg; Bronze 3104 CPU @ 1.70GHz is used for U50 tests.

:pushpin: **Note:** Performance numbers are computed using end-to-end latency and it depends on input image resolution. So performance numbers can vary with different images  