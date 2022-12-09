# Build flow  of Resnet50 example: 
:pushpin: **Note:** This application can be run only on **Versal VCK5000**

## Generate xclbin

##### **Note:** It is recommended to follow the build steps in sequence.

**VCK5000 xclbin generation**
* Download dpu xclbin build setup
    ```sh
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCVDX8H_vck5000
    wget https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_8pe_normal.tar.gz
    tar -xzvf DPUCVDX8H_8pe_normal.tar.gz
    cp -rf 8pe.mk ./DPUCVDX8H_8pe_normal/
    ```

* Open a linux terminal. Set the linux as Bash mode and execute following instructions.
    ```sh
    source < vitis-install-directory >/Vitis/2021.2/settings64.sh
    source < path-to-XRT-installation-directory >/setenv.sh
    export SDX_PLATFORM=< versal-vck5000-platform-path >/xilinx_vck5000_gen3x16_xdma_1_202110_1/xilinx_vck5000_gen3x16_xdma_1_202110_1.xpfm
    export DEVICE=$SDX_PLATFORM
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCVDX8H_vck5000
    bash -x run.sh
    ```

Note that 
- Generated xclbin will be here **${VAI_HOME}/examples/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCVDX8H_vck5000/work/2021.2/package.xclbin
- Build runtime is ~15 hours.
- Currently, the preprocess accelerator supports FHD image resolution. To change the maximum resolution of input image and other metrics, config params header file of the preprocess accelerator can be modified. Path: Vitis-AI/examples/Whole-App-Acceleration/plugins/blobfromimage/pl/xf_config_params.h

## Run Resnet50 Example
Please refer the instructions in [Setting up the system and running Resnet50 example on U50/U280/U200/VCK5000](../../README.md#setting-up-the-system-and-running-resnet50-example-on-u50u280u200vck5000) section to run the Resnet50 example.
