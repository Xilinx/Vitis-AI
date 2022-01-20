# Build flow  of ADAS example: 
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
    cd ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/build_flow/DPUCAHX8H_u50_u280
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
    cd ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/build_flow/DPUCAHX8H_u50_u280
    bash -x run_u280.sh
```
Note that 
- Generated xclbin will be here **${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/build_flow/DPUCAHX8H_u50_u280/bit_gen/u50.xclbin u280.xclbin**.
- Build runtime is ~9 hours for u50 and ~22 hours for u280.
- Currently, the preprocess accelerator supports FHD image resolution. To change the maximum resolution of input image and other metrics, config params header file of the preprocess accelerator can be modified. Path: Vitis-AI/Whole-App-Acceleration//plugins/blobfromimage/pl/xf_config_params.h

#### Setting up and running on U50 & U280
**Refer to [Setup Alveo Accelerator Card](../../../../../setup/alveo) to set up the Alveo Card.**

**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../../../README.md#Installation)**

* Install xclbin.
    * For U50 xclbin
	```
	sudo cp ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/build_flow/DPUCAHX8H_u50_u280/bit_gen/u50.xclbin /opt/xilinx/overlaybins/dpu.xclbin
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpu.xclbin
	```
    * For U280 xclbin
	```
	sudo cp ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/build_flow/DPUCAHX8H_u50_u280/bit_gen/u280.xclbin /opt/xilinx/overlaybins/dpu.xclbin
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpu.xclbin
	```
* Download and install ADAS detection model.
    ```
    mkdir -p ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/model_dir/model_u50-u50lv-u280
	wget https://www.xilinx.com/bin/public/openDownload?filename=yolov3_adas_pruned_0_9-u50-u50lv-u280-DPUCAHX8H-r1.4.1.tar.gz -O yolov3_adas_pruned_0_9-u50-u50lv-u280-DPUCAHX8H-r1.4.1.tar.gz
    tar -xzvf yolov3_adas_pruned_0_9-u50-u50lv-u280-DPUCAHX8H-r1.4.1.tar.gz -C ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/model_dir/model_u50-u50lv-u280
	```

* Download the images at https://cocodataset.org/#download or any other repositories and copy the images to ` ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/img` directory. In the following performance test we used COCO dataset. 

* Building ADAS detection application
	```
  cd ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection
  bash -x app_build.sh
	```

  If the compilation process does not report any error then the executable file `./bin/adas_detection.exe` is generated.    

* Run ADAS detection Example

  Following performance and functionality tests are indicated for **U50** platform.
  * Performance test with & without waa

    ```
    % export XLNX_ENABLE_FINGERPRINT_CHECK=0
    % ./app_test.sh --xmodel_file ./model_dir/model_u50-u50lv-u280/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel --image_dir ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/img --performance_diff

    Expect similar output:
	   Running Performance Diff: 

			Running Application with Software Preprocessing 

			E2E Performance: 66.69 fps
			Pre-process Latency: 4.59 fps
			Execution Latency: 3.47 fps
			Post-process Latency: 6.93 fps

			Running Application with Hardware Preprocessing 

			E2E Performance: 88.53 fps
			Pre-process Latency: 0.94 fps
			Execution Latency: 3.29 fps
			Post-process Latency: 7.06 fps

			The percentage improvement in throughput is 32.75 %
    ```

  * Functionality test with waa for a single image
    ```
    % ./app_test.sh --xmodel_file ./model_dir/model_u50-u50lv-u280/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel --image_dir ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/img --verbose

    Expect similar output:
		The Confidence Threshold used in this demo is 0.5
		Total number of images in the dataset is 1

		image name: image
		  xmin, ymin, xmax, ymax :12.3173 312.835 89.8562 364.71
		image name: image
		  xmin, ymin, xmax, ymax :108.805 328.219 144.592 353.546
		image name: image
		  xmin, ymin, xmax, ymax :83.6033 328.275 119.39 349.272
		image name: image
		  xmin, ymin, xmax, ymax :1.9616 324.017 21.5726 345.06
		image name: image
		  xmin, ymin, xmax, ymax :139.033 321.582 191.414 362.169
		image name: image
		  xmin, ymin, xmax, ymax :171.326 325.928 261.314 364.758
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
		  xmin, ymin, xmax, ymax :109.099 328.121 144.886 353.449
		image name: image
		  xmin, ymin, xmax, ymax :83.6033 328.426 119.39 349.423
		image name: image
		  xmin, ymin, xmax, ymax :173.604 326.026 258.141 364.855
		image name: image
		  xmin, ymin, xmax, ymax :139.336 321.671 191.717 362.259
		image name: image
		  xmin, ymin, xmax, ymax :1.87612 324.302 21.4872 345.345
    ```
	
  Following performance and functionality tests are indicated for U280 platform.
  * Performance test with & without waa

    ```
    % export XLNX_ENABLE_FINGERPRINT_CHECK=0
    % ./app_test.sh --xmodel_file ./model_dir/model_u50-u50lv-u280/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel --image_dir ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/img --performance_diff

    Expect similar output:
	  Running Performance Diff: 

			Running Application with Software Preprocessing 

			E2E Performance: 64.82 fps
			Pre-process Latency: 4.92 ms
			Execution Latency: 2.57 ms
			Post-process Latency: 7.93 ms

			Running Application with Hardware Preprocessing 

			E2E Performance: 81.99 fps
			Pre-process Latency: 0.81 ms
			Execution Latency: 2.49 ms
			Post-process Latency: 8.90 ms

			The percentage improvement in throughput is 26.49 %
    ```

  * Functionality test with waa for a single image
    ```
    % ./app_test.sh --xmodel_file ./model_dir/model_u50-u50lv-u280/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel --image_dir ${VAI_HOME}/Whole-App-Acceleration/apps/adas_detection/img --verbose

    Expect similar output:
		The Confidence Threshold used in this demo is 0.5
		Total number of images in the dataset is 1
		image name: image
		  xmin, ymin, xmax, ymax :12.3173 312.835 89.8562 364.71
		image name: image
		  xmin, ymin, xmax, ymax :108.805 328.219 144.592 353.546
		image name: image
		  xmin, ymin, xmax, ymax :83.6033 328.275 119.39 349.272
		image name: image
		  xmin, ymin, xmax, ymax :1.9616 324.017 21.5726 345.06
		image name: image
		  xmin, ymin, xmax, ymax :139.033 321.582 191.414 362.169
		image name: image
		  xmin, ymin, xmax, ymax :171.326 325.928 261.314 364.758    
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
		  xmin, ymin, xmax, ymax :109.099 328.121 144.886 353.449
		image name: image
		  xmin, ymin, xmax, ymax :83.6033 328.426 119.39 349.423
		image name: image
		  xmin, ymin, xmax, ymax :173.604 326.026 258.141 364.855
		image name: image
		  xmin, ymin, xmax, ymax :139.336 321.671 191.717 362.259
		image name: image
		  xmin, ymin, xmax, ymax :1.87612 324.302 21.4872 345.345
    ```
	
