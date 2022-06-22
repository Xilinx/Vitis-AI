# Build flow of PSMNet example:

:pushpin: **Note:** This application can be run only on VCK190 Board

## Generate Xclbin and SD card image

#### **Note:** It is recommended to follow the build steps in sequence.

Download and Unzip the dpu related aie and ip files.
     
The following tutorials assume that the Vitis and XRT environment variable is set as given below.

Download XVDPU-TRD Setup
```sh
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/psmnet/build_flow/DPUCVDX8G_vck190
    mkdir XVDPU-TRD
    wget https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8G.tar.gz
    tar -xzvf DPUCVDX8G.tar.gz --directory XVDPU-TRD
    cp -rf ./XVDPU-TRD/DPUCVDX8G/xvdpu_ip .
```

Open a linux terminal. Set the linux to Bash mode.

```sh
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/psmnet/build_flow/DPUCVDX8G_vck190
    source < vitis-install-directory >/Vitis/2022.1/settings64.sh
    source < path-to-XRT-installation-directory >/setenv.sh
    export PLATFORM=< path-to-vck190-platform >/xilinx_vck190_base_202210_1/xilinx_vck190_base_202210_1.xpfm
    export DEVICE=$PLATFORM
    export EDGE_COMMON_SW=< path-to-vck190-common-system >/versal/xilinx-versal-common-v2022.1/ 
    ./run.sh
```



**Note:** Generated SD card image will be at **${VAI_HOME}/examples/Whole-App-Acceleration/apps/psmnet/build_flow/DPUCVDX8G_vck190/vitis_prj/package_out/sd_card.img**.

> Flash the generated sd_card.img to an inserted SD card using Etcher software.

* To run the application, follow [psmnet/README.md](./../../README.md/#Setup)
