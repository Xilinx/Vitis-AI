# Pre-built flow of ADAS detection  example: 
:pushpin: **Note:** Pre-built flow for U50 card will be available in the upcoming release.
<details>
<summary>Click to expand expected instructions for pre-built flow:</summary>
:pushpin: **Note:** This application can be run only on **Alveo U50**

## Generate xclbin

##### **Note:** It is recommended to follow the build steps in sequence.

**U50 xclbin generation**
* Download dpu xo file
    ```sh
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/adas_detection/pre_built_flow/DPUCAHX8H_u50
    wget https://www.xilinx.com/bin/public/openDownload?filename=DPUCAHX8H.tar.gz
    tar -xzvf DPUCAHX8H.tar.gz
    cp -rf DPUCAHX8H/release_u50_xo release_u50_xo
    ```

* Download [Vitis-AI.2.5-WAA-pre-built-u50.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Vitis-AI.2.5-WAA-pre-built-u50.tar.gz). Untar the packet and copy the contents of `checkpoints` folder to `${VAI_HOME}/examples/Whole-App-Acceleration/apps/adas_detection/pre_built_flow/DPUCAHX8H_u50/checkpoints`.

* Open a linux terminal. Set the linux as Bash mode and execute following instructions.
    ```sh
    source < vitis-install-directory >/Vitis/2021.2/settings64.sh
    source < path-to-XRT-installation-directory >/setenv.sh
    export SDX_PLATFORM=< path-to-platform-directory >/xilinx_u50_gen3x4_xdma_2_202010_1/xilinx_u50_gen3x4_xdma_2_202010_1.xpfm
    export PLATFORM_REPO_PATHS=< path-to-platform-directory >
    export DEVICE=$SDX_PLATFORM
    cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/adas_detection/pre_built_flow/DPUCAHX8H_u50
    bash -x run_u50.sh
    ```
Note that 
- Generated xclbin will be here **${VAI_HOME}/examples/Whole-App-Acceleration/apps/adas_detection/pre_built_flow/DPUCAHX8H_u50/bit_gen/u50.xclbin u280.xclbin**.
- Build runtime is ~9 hours for u50.
- Currently, the preprocess accelerator supports FHD image resolution. To change the maximum resolution of input image and other metrics, config params header file of the preprocess accelerator can be modified. Path: Vitis-AI/examples/Whole-App-Acceleration//plugins/blobfromimage/pl/xf_config_params.h

## Run ADAS detection Example
Please refer the instructions in [Setting up the system and running ADAS detection example on U50/U280/U200](../../README.md#setting-up-the-system-and-running-adas-detection-example-on-u50u280u200) section to run the ADAS detection example.
</details>
