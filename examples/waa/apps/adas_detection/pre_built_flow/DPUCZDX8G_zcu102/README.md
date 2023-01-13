# Pre-built flow  of ADAS detection example: 
:pushpin: **Note:** This application can be run only on **Zynq UltraScale+ ZCU102**

## Generate xclbin and SD card image

##### **Note:** It is recommended to follow the build steps in sequence.

**ZCU102 xclbin and SD card files generation**
* Download and unzip [mpsoc common system](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zynqmp-common-v2022.1.tar.gz) & [zcu102 base platform](https://www.xilinx.com/member/forms/download/design-license-zcu102-base.html?filename=xilinx_zcu102_base_202210_1.zip) package.

    Please note that Xilinx account sign-in may be required to download some of the above files.

* The following tutorials assume that the Vitis and XRT environment variable is set as given below.

* Download DPU-TRD setup
    ```sh
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/adas_detection/pre_built_flow/DPUCZDX8G_zcu102
    mkdir DPU-TRD
    wget https://www.xilinx.com/bin/public/openDownload?filename=DPUCZDX8G.tar.gz
    tar -xzvf DPUCZDX8G.tar.gz --directory DPU-TRD
    ```

* Download [Vitis-AI.2.5-WAA-pre-built-zcu102.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Vitis-AI.2.5-WAA-pre-built-zcu102.tar.gz). Untar the packet and copy the contents of `checkpoints` folder to `${VAI_HOME}/examples/Whole-App-Acceleration/apps/adas_detection/pre_built_flow/DPUCZDX8G_zcu102/checkpoints`.

* Open a linux terminal. Set the linux as Bash mode and execute following instructions.
    ```sh
    source < vitis-install-directory >/Vitis/2022.1/settings64.sh
    source < path-to-XRT-installation-directory >/setenv.sh
    gunzip < mpsoc-common-system >/xilinx-zynqmp-common-v2022.1/rootfs.tar.gz
    export EDGE_COMMON_SW=< mpsoc-common-system >/xilinx-zynqmp-common-v2022.1
    export SDX_PLATFORM=< zcu102-base-platform-path >/xilinx_zcu102_base_202210_1/xilinx_zcu102_base_202210_1.xpfm
    export DEVICE=$SDX_PLATFORM
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/adas_detection/pre_built_flow/DPUCZDX8G_zcu102
    bash -x run.sh
    ```

Note that 
- Generated SD card image will be here **${VAI_HOME}/examples/Whole-App-Acceleration/apps/adas_detection/pre_built_flow/DPUCZDX8G_zcu102/binary_container_1/sd_card.img**.
- The default setting of DPU is **B4096** with RAM_USAGE_LOW, CHANNEL_AUGMENTATION_ENABLE, DWCV_ENABLE, POOL_AVG_ENABLE, RELU_LEAKYRELU_RELU6, Softmax. 
- Build runtime is ~1.25 hours.
- Currently, the preprocess accelerator supports FHD image resolution. To change the maximum resolution of input image and other metrics, config params header file of the preprocess accelerator can be modified. Path: `Vitis-AI/examples/Whole-App-Acceleration/plugins/blobfromimage/pl/xf_config_params.h`

## Run ADAS detection Example
Please refer the instructions in [Setting up the system and running ADAS detection example on ZCU102/ZCU104/VCK190](../../README.md#setting-up-the-system-and-running-adas-detection-example-on-zcu102zcu104vck190) section to run the ADAS detection example.
