# Build flow  of Resnet50 example: 
:pushpin: **Note:** This application can be run only on **Alveo U50 & U280**

## Generate xclbin

##### **Note:** It is recommended to follow the build steps in sequence.

**U50 xclbin generation**
* Download dpu xo file
    ```sh
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCAHX8H_u50_u280   
    wget https://www.xilinx.com/bin/public/openDownload?filename=DPUCAHX8H.tar.gz
    tar -xzvf DPUCAHX8H.tar.gz
    cp -rf DPUCAHX8H/release_u50_xo release_u50_xo
    ```

* Open a linux terminal. Set the linux as Bash mode and execute following instructions.
    ```sh
    source < vitis-install-directory >/Vitis/2021.2/settings64.sh
    source < path-to-XRT-installation-directory >/setenv.sh
    export SDX_PLATFORM=< path-to-platform-directory >/xilinx_u50_gen3x4_xdma_2_202010_1/xilinx_u50_gen3x4_xdma_2_202010_1.xpfm
    export PLATFORM_REPO_PATHS=< path-to-platform-directory >
    export DEVICE=$SDX_PLATFORM
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCAHX8H_u50_u280
    bash -x run_u50.sh
    ```

**U280 xclbin generation**
* Download dpu xo file
    ```sh
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCAHX8H_u50_u280
    wget https://www.xilinx.com/bin/public/openDownload?filename=DPUCAHX8H.tar.gz
    tar -xzvf DPUCAHX8H.tar.gz
    cp -rf DPUCAHX8H/release_u280_xo release_u280_xo
    ```

* Open a linux terminal. Set the linux as Bash mode and execute following instructions.
    ```sh
    source < vitis-install-directory >/Vitis/2021.2/settings64.sh
    source < path-to-XRT-installation-directory >/setenv.sh
    export SDX_PLATFORM=< path-to-platform-directory >/xilinx_u280_xdma_201920_3/xilinx_u280_xdma_201920_3.xpfm
    export PLATFORM_REPO_PATHS=< path-to-platform-directory >
    export DEVICE=$SDX_PLATFORM
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCAHX8H_u50_u280
    bash -x run_u280.sh
    ```
Note that 
- Generated xclbin will be here **${VAI_HOME}/examples/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCAHX8H_u50_u280/bit_gen/u50.xclbin u280.xclbin**.
- Build runtime is ~9 hours for u50 and ~22 hours for u280.
- Currently, the preprocess accelerator supports FHD image resolution. To change the maximum resolution of input image and other metrics, config params header file of the preprocess accelerator can be modified. Path: Vitis-AI/examples/Whole-App-Acceleration//plugins/blobfromimage/pl/xf_config_params.h

## Run Resnet50 Example
Please refer the instructions in [Setting up the system and running Resnet50 example on U50/U280/U200/VCK5000](../../README.md#setting-up-the-system-and-running-resnet50-example-on-u50u280u200vck5000) section to run the Resnet50 example.
