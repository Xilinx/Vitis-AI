# Classification example: Alveo U50 TRD run using Pre-processor files & pre-built DPU

## Build and run the application

### 1. Generate xclbin file
The following tutorials assume that the $TRD_HOME environment variable is set as given below.

```
%export TRD_HOME =< Vitis-AI-path >/dsa/WAA-TRD
```

###### **Note:** It is recommended to follow the build steps in sequence.

We need install the Vitis Core Development Environment.

Download [Vitis-AI.1.4-WAA-TRD.bin.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Vitis-AI.1.4-WAA-TRD.bin.tar.gz). Untar the packet and copy `bin` folder to `Vitis-AI/dsa/WAA-TRD/`. 

Open a linux terminal. Set the linux as Bash mode and execute following instructions.

```
% cd $TRD_HOME/proj/pre-built/classification-pre_DPUv3e
% source < vitis-install-directory >/Vitis/2020.2/settings64.sh
% source < path-to-XRT-installation-directory >/setup.sh
% export PLATFORM_REPO_PATHS=`readlink -f ../../../bin`
% export SDX_PLATFORM=xilinx_u50_gen3x4_xdma_2_202010_1
% ./run.sh
```
Note that 
- Generated xclbin will be here **$TRD_HOME/proj/pre-built/classification-pre_DPUv3e/_x_output/dpu.xclbin**.
- Build runtime is ~2.5 hours.

### 2. Setting Up the Target Alveo U50
**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../../../README.md#Installation)**

* Follow the steps mentioned [here](../../../../../setup/alveo/README.md) to setup the target. 

* Update xclbin & hbm address assignment file

	```
	  sudo cp ${VAI_HOME}/dsa/WAA-TRD/proj/pre-built/classification-pre_DPUv3e/_x_output/dpu.xclbin /usr/lib/dpu.xclbin
	  sudo cp ${VAI_HOME}/dsa/WAA-TRD/proj/pre-built/classification-pre_DPUv3e/hbm_address_assignment.txt /usr/lib/
	```	

* To download and install `resnet50` model:
	```
	  mkdir -p ${VAI_HOME}/dsa/WAA-TRD/app/resnet50/model
	  cd ${VAI_HOME}/dsa/WAA-TRD/app/resnet50/model
	  wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-u50-r1.3.0.tar.gz -O resnet50-u50-r1.3.0.tar.gz
	```	
	* Install the model package.


	```
	  tar -xzvf resnet50-u50-r1.3.0.tar.gz
	  sudo mkdir -p /usr/share/vitis_ai_library/models
	  sudo cp resnet50 /usr/share/vitis_ai_library/models -r
	```

* Download test images

    Download the images at http://image-net.org/download-images and copy 1000 images to `Vitis-AI/dsa/WAA-TRD/app/resnet50/img` 

### 3. Compile & run the application on Alveo U50

```
% cd ${VAI_HOME}/dsa/WAA-TRD/app/resnet50
%./build.sh
% #run with waa
%./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel 1 0

Expect: 
Image : ./img/bellpeppe-994958.JPEG
top[0] prob = 0.990457  name = bell pepper
top[1] prob = 0.004048  name = acorn squash
top[2] prob = 0.002455  name = cucumber, cuke
top[3] prob = 0.000903  name = zucchini, courgette
top[4] prob = 0.000703  name = strawberry

``

% #run without waa
%./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel 0 0

Expect: 
Image : ./img/bellpeppe-994958.JPEG
top[0] prob = 0.992920  name = bell pepper
top[1] prob = 0.003160  name = strawberry
top[2] prob = 0.001493  name = cucumber, cuke
top[3] prob = 0.000705  name = acorn squash
top[4] prob = 0.000428  name = zucchini, courgette`
