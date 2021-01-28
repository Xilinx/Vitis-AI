# Classification example: Alveo U50 TRD run using Pre-processor files & pre-built DPU

## Build and run the application

### 1. Generate xclbin file
The following tutorials assume that the $TRD_HOME environment variable is set as given below.

```
%export TRD_HOME =< Vitis-AI-path >/WAA-TRD
```

###### **Note:** It is recommended to follow the build steps in sequence.

We need install the Vitis Core Development Environment.

Download [Vitis-AI.1.3.1-WAA-TRD.bin.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Vitis-AI.1.3.1-WAA-TRD.bin.tar.gz). Untar the packet and copy `bin` folder to `Vitis-AI/dsa/WAA-TRD/`. 
Note that for bash, Vitis-AI.1.3.1-WAA-TRD.bin.tar.gz file can be obtained from here `/wrk/acceleration/users/maheshm/publicDownloadrepo/`

The following tutorials assume that the Vitis and XRT environment variable is set as given below.

Open a linux terminal. Set the linux as Bash mode.

```
% source < vitis-install-directory >/Vitis/2020.2/settings64.sh
% source < part-to-XRT-installation-directory >/setup.sh
% export SDX_PLATFORM=< alveo-u50-platform-path >/xilinx_u50_gen3x4_xdma_2_202010_1/xilinx_u50_gen3x4_xdma_2_202010_1.xpfm
% cd $TRD_HOME/proj/pre-built/classification-pre_DPUv3e
% ./run.sh
```
Note that 
- Generated xclbin will be here **$TRD_HOME/proj/pre-built/classification-pre_DPUv3e/_x_output_noKernFreq/dpu.xclbin**.
- Build runtime is ~1.5 hours.

### 2. Setting Up the Target Alveo U50
**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment**

* Follow the steps mentioned [here](../../../setup/alveo/u50_u50lv_u280/README.md) to setup the target. 

* Update xclbin file

	```
	  sudo cp /workspace/dsa/WAA-TRD/proj/pre-built/classification-pre_DPUv3e/_x_output_noKernFreq/dpu.xclbin /usr/lib/dpu.xclbin
	```	

* To download and install `resnet50` model:
	```
	  cd ${VAI_ALVEO_ROOT}/..
	  wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-u50-r1.3.0.tar.gz -O resnet50-u50-r1.3.0.tar.gz
	```	
	* Install the model package.


	```
	  tar -xzvf resnet50-u50-r1.3.0.tar.gz
	  sudo mkdir -p /usr/share/vitis_ai_library/models
	  sudo cp resnet50 /usr/share/vitis_ai_library/models -r
	```

* Download test images

    Download the images at http://image-net.org/download-images and copy 1000 images to `Vitis-AI/demo/Whole-App-Acceleration/resnet50_mt_py_waa/images` 
	
### Compile & run the application on Alveo U50

  	```
  	cd /workspace/dsa/WAA-TRD/app/resnet50_waa
	./build.sh
	./resnet50_waa /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
  	```


