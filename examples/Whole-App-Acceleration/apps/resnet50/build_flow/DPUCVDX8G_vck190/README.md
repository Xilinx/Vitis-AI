# Build flow of Resnet50 example: 
:pushpin: **Note:** This application can be run only on **Versal VCK190** production board.

## Generate xclbin and SD card image

##### **Note:** It is recommended to follow the build steps in sequence.

**VCK190 xclbin and SD card files generation**
* Download and extract [versal common system](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-versal-common-v2022.1_04191534.tar.gz) & [vck190 base platform](https://www.xilinx.com/member/forms/download/design-license-vck190-base-xef.html?filename=xilinx_vck190_base_202210_1.zip) packages.

    Please note that Xilinx account sign-in may be required to download some of the above files.

* The following tutorials assume that the Vitis and XRT environment variable is set as given below.

* Download XVDPU-TRD setup
    ```sh
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCVDX8G_vck190
    mkdir XVDPU-TRD
    wget https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8G.tar.gz
    tar -xzvf DPUCVDX8G.tar.gz --directory XVDPU-TRD
    cp -rf ./XVDPU-TRD/DPUCVDX8G/xvdpu_ip .
    ```

* Open a linux terminal. Set the linux as Bash mode and execute following instructions.
    ```sh
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCVDX8G_vck190
    source < vitis-install-directory >/Vitis/2022.1/settings64.sh
    source < path-to-XRT-installation-directory >/setenv.sh
    export EDGE_COMMON_SW=<path-to-versal_common-system>versal/xilinx-versal-common-v2022.1
    export PLATFORM=<path-to-platform-directory>/xilinx_vck190_base_202210_1/xilinx_vck190_base_202210_1.xpfm
    export DEVICE=$PLATFORM
    bash -x run.sh
    ```
Note that 
- Generated SD card image will be here `resnet50/build_flow/DPUCVDX8G_vck190/vitis_prj/package_out/sd_card.img`.
- Build runtime is ~3 hours.

## Run Resnet50 Example
Please refer the instructions in [Setting up the system and running Resnet50 example on ZCU102/ZCU104/VCK190](../../README.md#setting-up-the-system-and-running-resnet50-example-on-zcu102zcu104vck190) section to run the Resnet50 example.
