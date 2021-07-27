# Classification example: Alveo U200 run using Pre-processor & DPU

## Build and run the application

### 1. Generate xclbin file
The following tutorials assume that the $TRD_HOME environment variable is set as given below.

```
%export TRD_HOME =< Vitis-AI-path >/dsa/WAA-TRD
```

###### **Note:** It is recommended to follow the build steps in sequence.

We need install the Vitis Core Development Environment.

Open a linux terminal. Set the linux as Bash mode and execute following instructions.

```
% cd $TRD_HOME/proj/build/classification-pre_DPUv3int8
% source < vitis-install-directory >/Vitis/2021.1/settings64.sh
% source < path-to-XRT-installation-directory >/setup.sh
% export SDX_PLATFORM=< path-to-platform-directory >/xilinx_u200_gen3x16_xdma_1_202110_1/xilinx_u200_gen3x16_xdma_1_202110_1.xpfm
% ./build_classification_pre_int8.sh
```
Note that 
- Generated xclbin will be here **$TRD_HOME/proj/build/classification-pre_DPUv3int8/outputs/xclbin/dpdpuv3_wrapper.hw.xilinx_u200_gen3x16_xdma_1_202110_1.xclbin**.
- Build runtime is ~20 hours.

### 2. Setting Up the Target Alveo U200
**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../../../README.md#Installation)**

* Install xclbin.

	```
	sudo mkdir -p /opt/xilinx/overlaybins/dpuv3int8_trd
	sudo cp ${VAI_HOME}/dsa/WAA-TRD/proj/build/classification-pre_DPUv3int8/outputs/xclbin/dpdpuv3_wrapper.hw.xilinx_u200_gen3x16_xdma_1_202110_1.xclbin /opt/xilinx/overlaybins/dpuv3int8_trd/dpu.xclbin
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpuv3int8_trd
	```

* Download and install `resnet_v1_50_tf` model:

	```
	cd ${VAI_HOME}/dsa/WAA-TRD/app/resnet50_int8
	wget https://www.xilinx.com/bin/public/openDownload?filename=resnet_v1_50_tf-u200-u250-r1.4.0.tar.gz -O resnet_v1_50_tf-u200-u250-r1.4.0.tar.gz
	tar -xzvf resnet_v1_50_tf-u200-u250-r1.4.0.tar.gz
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


### Building and running the application on Alveo U200
* Build
	```
	cd ${VAI_HOME}/dsa/WAA-TRD/app/resnet50_int8
	bash -x build.sh
	```
* Run classification without waa
	```
	./resnet50_int8 resnet_v1_50_tf/resnet_v1_50_tf.xmodel ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 0 0
	```
* Run classification with waa
	```
	./resnet50_int8 resnet_v1_50_tf/resnet_v1_50_tf.xmodel ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 1 0
	```
