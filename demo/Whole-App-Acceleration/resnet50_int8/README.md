## Classification
:pushpin: **Note:** This application can be run only on Alveo-U200 platform.

## Table of Contents

- [Introduction](#Introduction)
- [Setting Up and Running on Alveo U200](#Setting-Up-and-Running-on-Alveo-U200)
    - [Setting Up the Target](#Setting-Up-the-Target-Alveo-U200)
    - [Building and running the application](#Building-and-running-the-application-on-Alveo-U200)
- [Performance](#Performance)    

## Introduction
Currently, applications accelerating pre-processing for classification networks (Resnet-50) is provided and can only run on Alveo-U200 platform (device part xcu200-fsgd2104-2-e). In this application, software JPEG decoder is used for loading input image. JPEG decoder transfer input image data to pre-processing kernel and the pre-processed data is transferred to the ML accelerator. Below image shows the inference pipeline.

<div align="center">
  <img width="75%" height="75%" src="../doc_images/block_dia_classification.PNG">
</div>

## Setting Up and Running on Alveo U200

### Setting Up the Target Alveo U200
**Refer to [Setup Alveo Accelerator Card](../../../setup/alveo) to set up the Alveo Card.**

**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../README.md#Installation)**

* Download the [dpuv3int8_xclbins_1_4_0](https://www.xilinx.com/bin/public/openDownload?filename=dpuv3int8_xclbins_1_4_0.tar.gz) xclbin tar and install xclbin.

	```
	sudo tar -xzvf dpuv3int8_xclbins_1_4_0.tar.gz --directory /
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpuv3int8/waa/classification
	```

* Download and install `resnet_v1_50_tf` model:
	```
	cd ${VAI_HOME}/demo/Whole-App-Acceleration/resnet50_int8
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
	cd ${VAI_HOME}/demo/Whole-App-Acceleration/resnet50_int8
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

* Run below for profiling
	```
	./resnet50_int8 resnet_v1_50_tf/resnet_v1_50_tf.xmodel ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min <0 for sw pre / 1 for hw pre> 1
	```

## Performance
Below table shows the comparison of performance achieved by accelerating the pre-processing pipeline on FPGA.
For `Resnet-50`, the performance numbers are achieved by running 1K images randomly picked from ImageNet dataset.
:pushpin: **Note:** The performance numbers doesn't include JPEG decoder.

FPGA: Alveo-U200

<table style="undefined;table-layout: fixed; width: 534px">
<colgroup>
<col style="width: 119px">
<col style="width: 136px">
<col style="width: 145px">
<col style="width: 134px">
</colgroup>
  <tr>
    <th rowspan="2">Network</th>
    <th colspan="2">E2E Throughput (fps)</th>
    <th rowspan="2"><span style="font-weight:bold">Percentage improvement</span></th>
  </tr>
  <tr>
    <td>with software Pre-processing</td>
    <td>with hardware Pre-processing</td>
  </tr>

  <tr>
    <td>Resnet-50</td>
    <td>156.5</td>
    <td>214.9</td>
    <td>37.3%</td>
  </tr>


</table>

**Note that Performance numbers are computed using end-to-end latency and it depends on input image resolution. So performance numbers can vary with different images**  
