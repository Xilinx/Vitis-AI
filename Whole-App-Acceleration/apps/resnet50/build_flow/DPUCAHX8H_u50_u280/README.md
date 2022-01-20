# Build flow  of Resnet50 example: 
:pushpin: **Note:** This application can be run only on **Alveo U50 & U280**

## Generate xclbin

###### **Note:** It is recommended to follow the build steps in sequence.

* U50 xclbin generation
   Open a linux terminal. Set the linux as Bash mode and execute following instructions.
```
    source < vitis-install-directory >/Vitis/2021.2/settings64.sh
    source < path-to-XRT-installation-directory >/setup.sh
    export SDX_PLATFORM=< path-to-platform-directory >/xilinx_u50_gen3x4_xdma_2_202010_1/xilinx_u50_gen3x4_xdma_2_202010_1.xpfm
    export PLATFORM_REPO_PATHS=< path-to-platform-directory >
    export DEVICE=$SDX_PLATFORM
    cd ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCAHX8H_u50_u280
    bash -x run_u50.sh
```

* U280 xclbin generation
   Open a linux terminal. Set the linux as Bash mode and execute following instructions.
```
    source < vitis-install-directory >/Vitis/2021.2/settings64.sh
    source < path-to-XRT-installation-directory >/setup.sh
    export SDX_PLATFORM=< path-to-platform-directory >/xilinx_u280_xdma_201920_3/xilinx_u280_xdma_201920_3.xpfm
    export PLATFORM_REPO_PATHS=< path-to-platform-directory >
    export DEVICE=$SDX_PLATFORM
    cd ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCAHX8H_u50_u280
    bash -x run_u280.sh
```
Note that 
- Generated xclbin will be here **${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCAHX8H_u50_u280/bit_gen/u50.xclbin u280.xclbin**.
- Build runtime is ~20 hours.
- Currently, the preprocess accelerator supports FHD image resolution. To change the maximum resolution of input image and other metrics, config params header file of the preprocess accelerator can be modified. Path: Vitis-AI/Whole-App-Acceleration//plugins/blobfromimage/pl/xf_config_params.h

#### Setting up and running on U50 & U280
**Refer to [Setup Alveo Accelerator Card](../../../../../setup/alveo) to set up the Alveo Card.**

**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../../../README.md#Installation)**

* Install xclbin.
    * For U50 xclbin
	```
	sudo cp ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCAHX8H_u50_u280/bit_gen/u50.xclbin /opt/xilinx/overlaybins/dpu.xclbin
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpu.xclbin
	```
    * For **U280** xclbin
	```
	sudo cp ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCAHX8H_u50_u280/bit_gen/u280.xclbin /opt/xilinx/overlaybins/dpu.xclbin
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpu.xclbin
	```
* Download and install resnet50 model.
    ```
    mkdir -p ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/model_dir
    wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-u50-u50lv-u280-DPUCAHX8H-r1.4.1.tar.gz -O resnet50-u50-u50lv-u280-DPUCAHX8H-r1.4.1.tar.gz
    tar -xzvf resnet50-u50-u50lv-u280-DPUCAHX8H-r1.4.1.tar.gz -C ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/model_dir
	```

* Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012/) using [Collective Knowledge (CK)](https://github.com/ctuning).

	```
  # Activate conda env
  conda activate vitis-ai-caffe
  python -m ck pull repo:ck-env
  python -m ck install package:imagenet-2012-val-min

  # We don't need conda env for running examples with this DPU
  conda deactivate
	```
  :pushpin: **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

* Building Resnet50 application
	```
  cd ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50
  bash -x app_build.sh
	```

  If the compilation process does not report any error then the executable file `./bin/resnet50.exe` is generated.    

* Run Resnet50 Example
  * Performance test with & without waa

    ```
    % export XLNX_ENABLE_FINGERPRINT_CHECK=0
    % ./app_test.sh --xmodel_file ./model_dir/resnet50/resnet50.xmodel --image_dir ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min --performance_diff

    Expect similar output:
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

  * Functionality test with waa
    ```
    % ./app_test.sh --xmodel_file ./model_dir/resnet50/resnet50.xmodel --image_dir ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min --verbose

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

  * Functionality test without waa for single image
    ```
    % ./app_test.sh --xmodel_file ./model_dir/resnet50/resnet50.xmodel --image_dir ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min --verbose --use_sw_pre_proc

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
