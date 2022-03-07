# Build flow  of ADAS example: 
:pushpin: **Note:** This application can be run only on **Alveo U200**

## Generate xclbin

###### **Note:** It is recommended to follow the build steps in sequence.

* U200 xclbin generation
	Open a linux terminal. Set the linux as Bash mode and execute following instructions.
```
    source < vitis-install-directory >/Vitis/2021.2/settings64.sh
    source < path-to-XRT-installation-directory >/setup.sh
    export SDX_PLATFORM=< alveo-u50-platform-path >/xilinx_u200_gen3x16_xdma_1_202110_1/xilinx_u200_gen3x16_xdma_1_202110_1.xpfm
    export DEVICE=$SDX_PLATFORM
    cd ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/build_flow/DPUCADF8H_u200
    bash -x run.sh
```	

Note that 
- Generated xclbin will be here **${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/build_flow/DPUCADF8H_u200/outputs/xclbin/dpdpuv3_wrapper.hw.xilinx_u200_gen3x16_xdma_1_202110_1.xclbin**.
- Build runtime is ~18.25 hours.
- Currently, the preprocess accelerator supports FHD image resolution. To change the maximum resolution of input image and other metrics, config params header file of the preprocess accelerator can be modified. Path: Vitis-AI/Whole-App-Acceleration//plugins/blobfromimage/pl/xf_config_params.h

#### Setting up and running on U200
**Refer to [Setup Alveo Accelerator Card](../../../../../setup/alveo) to set up the Alveo Card.**

**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../../../README.md#Installation)**

* Install xclbin.

	```
	sudo cp ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/build_flow/DPUCADF8H_u200/outputs/xclbin/dpdpuv3_wrapper.hw.xilinx_u200_gen3x16_xdma_1_202110_1.xclbin /opt/xilinx/overlaybins/dpu.xclbin
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpu.xclbin
	```

* Download and install ADAS detection model:

	```
    mkdir -p ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/model_dir/model_u200
	wget https://www.xilinx.com/bin/public/openDownload?filename=yolov3_adas_pruned_0_9-u200-u250-r1.4.0.tar.gz -O yolov3_adas_pruned_0_9-u200-u250-r1.4.0.tar.gz
    tar -xzvf yolov3_adas_pruned_0_9-u200-u250-r1.4.0.tar.gz -C ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/model_dir/model_u200
	```
* Download the images at https://cocodataset.org/#download or any other repositories and copy the images to ` ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/img` directory. In the following performance test we used COCO dataset. 

* Building ADAS detection application
	```
  cd ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection
  bash -x app_build.sh
	```

  If the compilation process does not report any error then the executable file `./bin/adas_detection.exe` is generated.    

* Run ADAS detection Example

  Following performance and functionality tests are indicated for U50 platform.
  * Performance test with & without waa

    ```
    % export XLNX_ENABLE_FINGERPRINT_CHECK=0
    % ./app_test.sh --xmodel_file ./model_dir/model_u50-u50lv-u280/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel --image_dir ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/img --performance_diff

    Expect similar output:
		Running Performance Diff: 

			Running Application with Software Preprocessing 

			E2E Performance: 37.17 fps
			Pre-process Latency: 6.16 ms
			Execution Latency: 11.61 ms
			Post-process Latency: 9.13 ms

			Running Application with Hardware Preprocessing 

			E2E Performance: 45.95 fps
			Pre-process Latency: 0.95 ms
			Execution Latency: 11.65 ms
			Post-process Latency: 9.16 ms

			The percentage improvement in throughput is 23.62 %
    ```

  * Functionality test with waa for a single image
    ```
    % ./app_test.sh --xmodel_file ./model_dir/model_u50-u50lv-u280/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel --image_dir ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/img --verbose

    Expect similar output:
		The Confidence Threshold used in this demo is 0.5
		Total number of images in the dataset is 1
		Currently, u200 doesnot support zero copy. Running without zero copy
		image name: image
		  xmin, ymin, xmax, ymax :12.3173 312.835 89.8562 364.71
		image name: image
		  xmin, ymin, xmax, ymax :109.889 328.219 143.508 353.546
		image name: image
		  xmin, ymin, xmax, ymax :83.6033 328.349 119.39 349.347
		image name: image
		  xmin, ymin, xmax, ymax :1.87612 324.208 21.4872 345.251
		image name: image
		  xmin, ymin, xmax, ymax :139.182 321.671 191.563 362.259
		image name: image
		  xmin, ymin, xmax, ymax :174.052 325.928 258.588 364.758
    ```

  * Functionality test without waa for a single image
    ```
    % ./app_test.sh --xmodel_file ./model_dir/model_u50-u50lv-u280/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel --image_dir ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/img --verbose --use_sw_pre_proc

    Expect similar output:
		The Confidence Threshold used in this demo is 0.5
		Total number of images in the dataset is 1
		image name: image
		  xmin, ymin, xmax, ymax :12.1682 313.02 89.707 364.895
		image name: image
		  xmin, ymin, xmax, ymax :109 328.219 144.787 353.546
		image name: image
		  xmin, ymin, xmax, ymax :83.7019 327.672 119.489 350.024
		image name: image
		  xmin, ymin, xmax, ymax :173.749 326.129 258.285 364.958
		image name: image
		  xmin, ymin, xmax, ymax :1.79266 324.208 21.4037 345.251
		image name: image
		  xmin, ymin, xmax, ymax :137.805 322.986 193.564 361.115
    ```
	