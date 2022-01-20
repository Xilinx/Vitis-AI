# Build flow  of Resnet50 example: 
:pushpin: **Note:** This application can be run only on Alveo **U200**

## Generate xclbin

###### **Note:** It is recommended to follow the build steps in sequence.

Open a linux terminal. Set the linux as Bash mode and execute following instructions.

```
% source < vitis-install-directory >/Vitis/2021.1/settings64.sh
% source < path-to-XRT-installation-directory >/setup.sh
% export SDX_PLATFORM=< path-to-platform-directory >/xilinx_u200_gen3x16_xdma_1_202110_1/xilinx_u200_gen3x16_xdma_1_202110_1.xpfm
% cd ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCADF8H_u200
% ./run.sh
```
Note that 
- Generated xclbin will be here **${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCADF8H_u200/outputs/xclbin/dpdpuv3_wrapper.hw.xilinx_u200_gen3x16_xdma_1_202110_1.xclbin**.
- Build runtime is ~20 hours.
- Currently, the preprocess accelerator supports FHD image resolution. To change the maximum resolution of input image and other metrics, config params header file of the preprocess accelerator can be modified. Path: Vitis-AI/Whole-App-Acceleration//plugins/blobfromimage/pl/xf_config_params.h

### Setting Up the Target Alveo U200
**Refer to [Setup Alveo Accelerator Card](../../../../../setup/alveo) to set up the Alveo Card.**

**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../../../README.md#Installation)**

* Install xclbin.

	```
	sudo cp ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCADF8H_u200/outputs/xclbin/dpdpuv3_wrapper.hw.xilinx_u200_gen3x16_xdma_1_202110_1.xclbin /opt/xilinx/overlaybins/dpu.xclbin
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpu.xclbin
	```

* Download and install resnet50 model:

	```
        mkdir -p ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/model_dir
	wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-u200-u250-r1.4.0.tar.gz -O resnet50-u200-u250-r1.4.0.tar.gz
        tar -xzvf resnet50-u200-u250-r1.4.0.tar.gz -C ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/model_dir
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

          E2E Performance: 149.68 fps
          Pre-process Latency: 2.86 ms
          Execution Latency: 3.40 ms
          Post-process Latency: 0.42 ms

          Running Application with Hardware Preprocessing 

          E2E Performance: 187.30 fps
          Pre-process Latency: 1.38 ms
          Execution Latency: 3.46 ms
          Post-process Latency: 0.49 ms

          The percentage improvement in throughput is 25.13 %

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
