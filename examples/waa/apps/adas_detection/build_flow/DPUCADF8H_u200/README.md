# Build flow  of ADAS detection example: 
:pushpin: **Note:** This application can be run only on **Alveo U200**

## Generate xclbin

##### **Note:** It is recommended to follow the build steps in sequence.

**U200 xclbin generation**
* Download dpu xo file
    ```sh
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/adas_detection/build_flow/DPUCADF8H_u200
    wget https://www.xilinx.com/bin/public/openDownload?filename=DPUCADF8H.tar.gz
    tar -xzvf DPUCADF8H.tar.gz
    mkdir xo_release
    cp -rf DPUCADF8H/* xo_release
    ```

* Open a linux terminal. Set the linux as Bash mode and execute following instructions.
    ```sh
    source < vitis-install-directory >/Vitis/2021.2/settings64.sh
    source < path-to-XRT-installation-directory >/setenv.sh
    export SDX_PLATFORM=< alveo-u200-platform-path >/xilinx_u200_gen3x16_xdma_1_202110_1/xilinx_u200_gen3x16_xdma_1_202110_1.xpfm
    export DEVICE=$SDX_PLATFORM
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/adas_detection/build_flow/DPUCADF8H_u200
    bash -x run.sh
    ```

Note that 
- Generated xclbin will be here **${VAI_HOME}/examples/Whole-App-Acceleration/apps/adas_detection/build_flow/DPUCADF8H_u200/outputs/xclbin/dpdpuv3_wrapper.hw.xilinx_u200_gen3x16_xdma_1_202110_1.xclbin**.
- Build runtime is ~18.25 hours.
- Currently, the preprocess accelerator supports FHD image resolution. To change the maximum resolution of input image and other metrics, config params header file of the preprocess accelerator can be modified. Path: Vitis-AI/examples/Whole-App-Acceleration//plugins/blobfromimage/pl/xf_config_params.h

## Run ADAS detection Example
Please refer the instructions in [Setting up the system and running ADAS detection example on U50/U280/U200](../../README.md#setting-up-the-system-and-running-adasdetection-example-on-u50u280u200) section to run the ADAS detection example.
