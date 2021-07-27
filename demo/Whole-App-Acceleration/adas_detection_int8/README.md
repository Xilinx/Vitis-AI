## ADAS detection
:pushpin: **Note:** This application can be run only on Alveo-U200 platform.

## Table of Contents

- [Introduction](#Introduction)
- [Setting Up and Running on Alveo U200](#Setting-Up-and-Running-on-Alveo-U200)
    - [Setting Up the Target](#Setting-Up-the-Target-Alveo-U200)
    - [Building and running the application](#Building-and-running-the-application-on-Alveo-U200)
- [Performance](#Performance)  

## Introduction

ADAS (Advanced Driver Assistance Systems) application
using YOLO-v3 network model is an example for object detection.
Accelerating pre-processing for YOLO-v3 is provided and can only run on Alveo-U200 platform (device part xcu200-fsgd2104-2-e). In this application, software JPEG decoder is used for loading input image. Three processes are created one for image loading and running pre-processing kernel ,one for running the ML accelerator and one for generating output image. JPEG decoder transfer input image data to pre-processing kernel and the pre-processed data is transferred to the ML accelerator over a queue. ML accelerator output will be transferred over queue to create output image. Below image shows the inference pipeline.

<div align="center">
  <img width="75%" height="75%" src="../doc_images/block_dia_adasdetection.PNG">
</div>

## Setting Up and Running on Alveo U200

### Setting Up the Target Alveo U200
**Refer to [Setup Alveo Accelerator Card](../../../setup/alveo) to set up the Alveo Card.**

**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../README.md#Installation)**

* Download the [dpuv3int8_xclbins_1_4_0](https://www.xilinx.com/bin/public/openDownload?filename=dpuv3int8_xclbins_1_4_0.tar.gz) xclbin tar and install xclbin.

	```
	sudo tar -xzvf dpuv3int8_xclbins_1_4_0.tar.gz --directory /
	export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpuv3int8/waa/detection
	```

* Download and install `yolov3_adas_pruned_0_9` model.
	```
  cd ${VAI_HOME}/demo/Whole-App-Acceleration/adas_detection_int8
  wget https://www.xilinx.com/bin/public/openDownload?filename=yolov3_adas_pruned_0_9-u200-u250-r1.4.0.tar.gz -O yolov3_adas_pruned_0_9-u200-u250-r1.4.0.tar.gz
  tar -xzvf yolov3_adas_pruned_0_9-u200-u250-r1.4.0.tar.gz
	```

* Download test images

  For adas_detection_int8 example, download the images at https://cocodataset.org/#download or  download using [Collective Knowledge (CK)](https://github.com/ctuning).
    ```
    # Activate conda env
    conda activate vitis-ai-caffe
    python -m ck pull repo:ck-env
  
    # Download COCO dataset (This may take a while as COCO val dataset is more than 6 GB in size)
    python -m ck install package:dataset-coco-2014-val

    # We don't need conda env for running examples with this DPU
    conda deactivate
    ```

:pushpin: **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

### Building and running the application on Alveo U200
* Build
    ```
    cd ${VAI_HOME}/demo/Whole-App-Acceleration/adas_detection_int8
    bash -x build.sh
    mkdir output #Will be written to the picture after processing
    ```
* Run adas_detection without waa
    ```
    ./adas_detection_int8 yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 0 <img dir>
    ```
* Run adas_detection with waa
    ```
    ./adas_detection_int8 yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 1 <img dir>
    ```

## Performance
Below table shows the comparison of latency achieved by acclerating the pre-processing pipeline on FPGA.
For `Adas Detection`, the performance numbers are achieved by running 1K images randomly picked from COCO dataset.

Network: YOLOv3 Adas Detection
<table style="undefined;table-layout: fixed; width: 534px">
<colgroup>
<col style="width: 119px">
<col style="width: 136px">
<col style="width: 145px">
<col style="width: 134px">
</colgroup>
  <tr>
    <th rowspan="2">FPGA</th>
    <th colspan="2">Latency (ms)</th>
    <th rowspan="2"><span style="font-weight:bold">Percentage improvement</span></th>
  </tr>
  <tr>
    <td>with software Pre-processing</td>
    <td>with hardware Pre-processing</td>
  </tr>



  <tr>
   <td>Alveo U200</td>
    <td>12.7</td>
    <td>2.1</td>
        <td>83.4%</td>
  </tr>
</table>

**Note that Performance numbers are computed using average latency and it depends on input image resolution. So performance numbers can vary with different images**   
