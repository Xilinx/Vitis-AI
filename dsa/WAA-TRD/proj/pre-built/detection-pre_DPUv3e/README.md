# Detection example: Alveo U50 TRD run using Pre-processor files & pre-built DPU

## Build and run the application

### 1. Generate xclbin file
The following tutorials assume that the $TRD_HOME environment variable is set as given below.

```
%export TRD_HOME =< Vitis-AI-path >/dsa/WAA-TRD
```

###### **Note:** It is recommended to follow the build steps in sequence.

We need install the Vitis Core Development Environment.

Download [Vitis-AI.1.4-WAA-TRD.bin.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Vitis-AI.1.4-WAA-TRD.bin.tar.gz). Untar the packet and copy `bin` folder to `Vitis-AI/dsa/WAA-TRD/`. 


Open a linux terminal. Set the linux as Bash mode and execute follwoing instructions.

```
% cd $TRD_HOME/proj/pre-built/detection-pre_DPUv3e
% source < vitis-install-directory >/Vitis/2020.2/settings64.sh
% source < path-to-XRT-installation-directory >/setup.sh
% export PLATFORM_REPO_PATHS=`readlink -f ../../../bin`
% export SDX_PLATFORM=xilinx_u50_gen3x4_xdma_2_202010_1
% ./run.sh
```
Note that 
- Generated xclbin will be here **$TRD_HOME/proj/pre-built/detection-pre_DPUv3e/_x_output/dpu.xclbin**.
- Build runtime is ~2.5 hours.

### 2. Setting Up the Target Alveo U50
**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../../../README.md#Installation)**

* Follow the steps mentioned [here](../../../../../setup/alveo/README.md) to setup the target. 

* Update xclbin and hbm address assignment file

	```
	  sudo cp ${VAI_HOME}/dsa/WAA-TRD/proj/pre-built/detection-pre_DPUv3e/_x_output/dpu.xclbin /usr/lib/dpu.xclbin
	  sudo cp ${VAI_HOME}/dsa/WAA-TRD/proj/pre-built/detection-pre_DPUv3e/hbm_address_assignment.txt /usr/lib/
	```	
* To download and install `adas detection` model:
	```
	  mkdir -p ${VAI_HOME}/dsa/WAA-TRD/app/adas_detection/model
	  cd ${VAI_HOME}/dsa/WAA-TRD/app/adas_detection/model
	  wget https://www.xilinx.com/bin/public/openDownload?filename=yolov3_adas_pruned_0_9-u50-r1.3.0.tar.gz -O yolov3_adas_pruned_0_9-u50-r1.3.0.tar.gz
	```	
* Install the model package.
	```
	  tar -xzvf yolov3_adas_pruned_0_9-u50-r1.3.0.tar.gz
	  sudo mkdir -p /usr/share/vitis_ai_library/models
	  sudo cp yolov3_adas_pruned_0_9 /usr/share/vitis_ai_library/models -r
	```
* Download test images	

  For adas_detection example, download the images at https://cocodataset.org/#download and copy the images to `Vitis-AI/dsa/WAA-TRD/app/adas_detection/data`

### 3. Compile & run the application on Alveo U50

```
% cd ${VAI_HOME}/dsa/WAA-TRD/app/adas_detection
% ./build.sh
% mkdir output
% ./adas_detection /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 1

Expect:
    Found Platform
    Platform Name: Xilinx
    INFO: Reading /usr/lib/dpu.xclbin
    Loading: '/usr/lib/dpu.xclbin'
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I0722 03:34:30.627436 111 main.cc:458] create running for subgraph: subgraph_layer0-conv
    Performance:45.7 FPS

```

