## Tensorflow SSD-Mobilenet Model
:pushpin: **Note:** This application can be run only on Alveo-U280 platform.

## Table of Contents

- [Introduction](#Introduction)
- [Set Up the target platform](#Setup)
- [Running the Application](#Running-the-Application)
- [Performance](#Performance)

## Introduction
The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. Accelerated pre-processing(resize, colour conversion, and normalization) and post-processing(Sort and NMS) for ssd-mobilenet is provided and can only run on U280 board. In this application, software JPEG decoder is used for loading input image. The pre-processed data is directly stored at ML accelerator input phsical address and post-process accelerator will directly read data from the ML accelerator output physical address. Hence avoided device to host data transfers.

## Setup
**Refer to [Setup Alveo Accelerator Card](../../../setup/alveo) to set up the Alveo Card.**

**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../README.md#Installation)**

### Data Preparation
- Download and extract coco datatset. (wget http://images.cocodataset.org/zips/val2017.zip)
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

### Download xclbin
- Download the [dpuv3me_xclbins_1_4_0](https://www.xilinx.com/bin/public/openDownload?filename=dpuv3me_xclbins_1_4_0.tar.gz) xclbin tar and install xclbin.
- `sudo tar -xzvf dpuv3me_xclbins_1_4_0.tar.gz --directory /`
- `export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpuv3me/waa/detection`

### Download model
- Download and extract model tar.
- `cd ${VAI_HOME}/demo/Whole-App-Acceleration/ssd_mobilenet_zero_copy/`
- `wget https://www.xilinx.com/bin/public/openDownload?filename=ssd_mobilenet_v1_coco_tf-u50-u50lv-u280-DPUCAHX8L-r1.4.0.tar.gz -O ssd_mobilenet_v1_coco_tf-u50-u50lv-u280-DPUCAHX8L-r1.4.0.tar.gz`
- `tar -xvf ssd_mobilenet_v1_coco_tf-u50-u50lv-u280-DPUCAHX8L-r1.4.0.tar.gz`

## Build the Application
- `make build && make -j`

## Running the Application using Hardware accelerated pre and post process
- `./app.exe ssd_mobilenet_v1_coco_tf/ssd_mobilenet_v1_coco_tf.prototxt ssd_mobilenet_v1_coco_tf/ssd_mobilenet_v1_coco_tf.xmodel <image directory> 1 1 0`

## Running the Application using Software pre and post process
- `./app.exe ssd_mobilenet_v1_coco_tf/ssd_mobilenet_v1_coco_tf.prototxt ssd_mobilenet_v1_coco_tf/ssd_mobilenet_v1_coco_tf.xmodel <image directory> 0 0 0`

## Profiling the Application
- `./app.exe ssd_mobilenet_v1_coco_tf/ssd_mobilenet_v1_coco_tf.prototxt ssd_mobilenet_v1_coco_tf/ssd_mobilenet_v1_coco_tf.xmodel <image directory> <sw/hw pre> <sw/hw post> 1`

## Detection Output
Detection outputs contains the lable, coordinates and confidence values for given input image.
Example:
```sh
Detection Output:
label, xmin, ymin, xmax, ymax, confidence : 1   506.328 169.578 632.734 386.739 0.867036
label, xmin, ymin, xmax, ymax, confidence : 1   8.35938 154.466 128.203 395.163 0.835484
label, xmin, ymin, xmax, ymax, confidence : 1   316.699 164.823 392.676 374.565 0.731059
```

### Performance:
Below table shows the comparison of througput (with out imread/jpeg-decoder) achieved by acclerating the pre-processing and post process on FPGA. 
For `SSD Mobilenet`, the performance numbers are achieved by running 5K images from COCO dataset.

Network: SSD Mobilenet
<table style="undefined;table-layout: fixed; width: 534px">
<colgroup>
<col style="width: 119px">
<col style="width: 136px">
<col style="width: 145px">
<col style="width: 134px">
</colgroup>
  <tr>
    <th rowspan="2">FPGA</th>
    <th colspan="2">E2E Throughput (fps)</th>
    <th rowspan="2"><span style="font-weight:bold">Percentage improvement in throughput</span></th>
  </tr>
  <tr>
    <td>with software Pre and post processing</td>
    <td>with hardware Pre and post processing</td>
  </tr>


  
  <tr>
   <td>Alveo-U280</td>
    <td>206.9</td>
    <td>360.5</td>
        <td>74.23%</td>
  </tr>

</table>

**Note that Performance numbers are computed using end-to-end latency and it depends on input image resolution. So performance numbers can vary with different images**
