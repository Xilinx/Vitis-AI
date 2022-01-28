# Build flow of PSMNet example:

:pushpin: **Note:** This application can be run only on VCK190 Board

## Generate SD card image

#### **Note:** It is recommended to follow the build steps in sequence.

The following tutorials assume that the Vitis and XRT environment variable is set as given below.

Open a linux terminal. Set the linux to Bash mode.

```sh
    source < vitis-install-directory >/Vitis/2021.1/settings64.sh
    source < path-to-XRT-installation-directory >/setup.sh
    cd ${VAI_HOME}/Whole-App-Acceleration/apps/psmnet/build_flow/DPUCVDX8G_vck190/
    wget https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8G_vck190_platform_2021.1.tar.gz -O DPUCVDX8G_vck190_platform_2021.1.tar.gz
    tar -xzvf DPUCVDX8G_vck190_platform_2021.1.tar.gz
    ./run.sh
```

**Note:** Generated SD card image will be at **${VAI_HOME}/Whole-App-Acceleration/apps/psmnet/build_flow/DPUCVDX8G_vck190/vitis_prj/package_out/sd_card.img**.

> Flash the generated sd_card.img to an inserted SD card using Etcher software.

* To run the application, follow [psmnet/README.md](./../../README.md/#Setup)
