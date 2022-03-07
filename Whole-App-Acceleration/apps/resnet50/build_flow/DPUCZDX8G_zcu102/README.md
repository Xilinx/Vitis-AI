# Build flow  of Resnet50 example: 
:pushpin: **Note:** This application can be run only on ZCU102

## Generate SD card image

###### **Note:** It is recommended to follow the build steps in sequence.

Download and unzip [mpsoc common system](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zynqmp-common-v2021.2.tar.gz) & [zcu102 base platform](https://www.xilinx.com/member/forms/download/design-license-zcu102-base.html?filename=xilinx_zcu102_base_202120_1.zip) package.

The following tutorials assume that the Vitis and XRT environment variable is set as given below.

Open a linux terminal. Set the linux as Bash mode.

```
    source < vitis-install-directory >/Vitis/2021.2/settings64.sh
    source < path-to-XRT-installation-directory >/setup.sh
    gunzip < mpsoc-common-system >/xilinx-zynqmp-common-v2021.2/rootfs.tar.gz
    export EDGE_COMMON_SW=< mpsoc-common-system >/xilinx-zynqmp-common-v2021.2 
    export SDX_PLATFORM=< zcu102-base-platform-path >/xilinx_zcu102_base_202120_1/xilinx_zcu102_base_202120_1.xpfm
    export DEVICE=$SDX_PLATFORM
    cd ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCZDX8G_zcu102
    bash -x run.sh
```

Note that 
- Generated SD card image will be here **${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCZDX8G_zcu102/binary_container_1/sd_card.img**.
- The default setting of DPU is **B4096** with RAM_USAGE_LOW, CHANNEL_AUGMENTATION_ENABLE, DWCV_ENABLE, POOL_AVG_ENABLE, RELU_LEAKYRELU_RELU6, Softmax. Modify the `${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCZDX8G_zcu102/dpu_conf.vh` file can change the default settings.
- Build runtime is ~1.5 hours
- Currently, the preprocess accelerator supports FHD image resolution. To change the maximum resolution of input image and other metrics, config params header file of the preprocess accelerator can be modified. Path: Vitis-AI/Whole-App-Acceleration//plugins/blobfromimage/pl/xf_config_params.h

## Installing board image
- Use Etcher software to burn the sd card image file onto the SD card.

## Setting up & running on ZCU102

### Installing Vitis AI Runtime on the Evaluation Board

- Download the [Vitis AI Runtime 2.0.0](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-2.0.0.tar.gz). 

- Untar the runtime packet and copy the following folder to the board using scp.
```
	tar -xzvf vitis-ai-runtime-2.0.0.tar.gz
	scp -r vitis-ai-runtime-2.0.0/2021.2/aarch64/centos root@IP_OF_BOARD:~/
```
- Install the Vitis AI Runtime on the evaluation board. Execute the following command.
```
	cd ~/centos
	bash setup.sh
```
### Cross-compile WAA-TRD example
* Download the [sdk-2021.2.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2021.2.0.0.sh)

* Install the cross-compilation system environment, follow the prompts to install. 

    **Please install it on your local host linux system, not in the docker system.**
    ```
    ./sdk-2021.2.0.0.sh
    ```
    Note that the `~/petalinux_sdk` path is recommended for the installation. Regardless of the path you choose for the installation, make sure the path has read-write permissions. 
Here we install it under `~/petalinux_sdk`.

* When the installation is complete, follow the prompts and execute the following command.
    ```
    . ~/petalinux_sdk/environment-setup-cortexa72-cortexa53-xilinx-linux
    ```
    Note that if you close the current terminal, you need to re-execute the above instructions in the new terminal interface.

* Download the [vitis_ai_2021.2-r2.0.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2021.2-r2.0.0.tar.gz) and install it to the petalinux system.
    ```
    tar -xzvf vitis_ai_2021.2-r2.0.0.tar.gz -C ~/petalinux_sdk/sysroots/cortexa72-cortexa53-xilinx-linux
    ```

* Cross compile `resnet50` example.
    ```
    cd  ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50
    bash -x app_build.sh
    ```
    If the compilation process does not report any error and the executable file `./bin/resnet50.exe` is generated , then the host environment is installed correctly.



### Download Model files for Resnet50

```
%	cd ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50
%	mkdir model_zcu102
%	cd model_zcu102
%	wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102_zcu104_kv260-r2.0.0.tar.gz -O resnet50-zcu102_zcu104_kv260-r2.0.0.tar.gz
%	tar -xzvf resnet50-zcu102_zcu104_kv260-r2.0.0.tar.gz
```

### Run Resnet50 Example
This part is about how to run the Resnet50 example on zcu102 board.

* Download the images at http://image-net.org/download-images and copy images to ` ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/img` directory 

* Copy following content of  ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50 directory to the BOOT partition of the SD Card.
    ```
        bin
        model_zcu102
        img
        app_test.sh
        word.txt
    ```


* Please insert SD_CARD on the ZCU102 board. After the linux boot, run:

* Performance test with & without waa

    ```
    % cd /media/sd-mmcblk0p1/
    % export XLNX_VART_FIRMWARE=/media/sd-mmcblk0p1/dpu.xclbin
    %
    % ./app_test.sh --xmodel_file ./model_zcu102/resnet50/resnet50.xmodel --image_dir ./img --performance_diff

    Expect similar output:
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

* Functionality test with waa
    ```
    % ./app_test.sh --xmodel_file ./model_zcu102/resnet50/resnet50.xmodel --image_dir ./img --verbose

    Expect similar output:
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I0712 10:16:33.656128  1587 main.cc:465] create running for subgraph: subgraph_conv1
    Number of images in the image directory is: 1
    top[0] prob = 0.829972  name = sea snake
    top[1] prob = 0.068128  name = hognose snake, puff adder, sand viper
    top[2] prob = 0.032181  name = water snake
    top[3] prob = 0.015201  name = horned viper, cerastes, sand viper, horned asp, Cerastes cornutus
    top[4] prob = 0.015201  name = American alligator, Alligator mississipiensis
    ```

* Functionality test without waa
    ```
    % ./app_test.sh --xmodel_file ./model_zcu102/resnet50/resnet50.xmodel --image_dir ./img --verbose --use_sw_pre_proc

    Expect similar output:
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I0712 10:16:42.329468  1612 main.cc:465] create running for subgraph: subgraph_conv1
    Number of images in the image directory is: 1
    top[0] prob = 0.808481  name = sea snake
    top[1] prob = 0.066364  name = hognose snake, puff adder, sand viper
    top[2] prob = 0.031348  name = water snake
    top[3] prob = 0.031348  name = American alligator, Alligator mississipiensis
    top[4] prob = 0.024414  name = African crocodile, Nile crocodile, Crocodylus niloticus
    ```