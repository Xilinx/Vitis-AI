# Accelerating Pre-processing for YOLO inference

This application  demonstrates the acceleration of pre-processing of inference of object detection networks like Yolo-v3 and Tiny Yolo-v3. Below block diagrams show various steps involved in the pre-processing and the blocks which are accelerated on hardware.

<div align="center">
  <img width="75%" height="75%" src="./doc_images/block_dia_sw_pp.PNG">
</div>

<div align="center">
  <img width="75%" height="75%" src="./doc_images/block_dia_hw_pp.PNG">
</div>

## Setup
```sh
# Activate Conda Environment
conda activate vitis-ai-caffe 
```
```sh
# Setup
source /workspace/alveo/overlaybins/setup.sh
```

## Running the Application
-  `cd ${VAI_ALVEO_ROOT}/apps/whole_app_acceleration/yolo`
- Use `detect.sh` file to run the application.
- Make sure to follow the steps [here](../../yolo/README.md#getting-coco-2014-validation-set-and-labels) to get COCO validation set and labels/ 

> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

### Examples

- Familiarize yourself with the script usage by 
```sh
./detect.sh -h
```
- To run yolo inference without JPEG decoder accelerator
```sh
 ./detect.sh
```

- To run tiny_yolo_v3 inference with JPEG decoder accelerator
```sh
 ./detect.sh -t test_detect_jpeg -m tiny_yolo_v3 --neth 608 --netw 608
```

> **Note:** Currently, JPEG decode accelerator doesn't support certain JPEG image types so a conversion script is run to convert the input images to supported format. Network dimensions of 608x608 are only supported.

### Performance

Below table shows the comparison of pre-processing execution times on CPU and FPGA and also the througput achieved by acclerating the pre-processing pipeline on FPGA. The performance numbers are achieved by running 5K images randomly picked from COCO dataset. The performance results may vary based on your system performance. 

CPU:  Intel(R) Core(TM) i7-4770 CPU @ 3.40GHz

FPGA: Alveo-U200


<table style="undefined;table-layout: fixed; width: 533px">
<colgroup>
<col style="width: 119px">
<col style="width: 136px">
<col style="width: 144px">
<col style="width: 134px">
</colgroup>
  <tr>
    <th rowspan="2">Network</th>
    <th colspan="2">E2E Throughput (fps)</th>
    <th rowspan="2"><span style="font-weight:bold">Percentage improvement in throughput</span></th>
  </tr>
  <tr>
    <td>with software Pre-processing</td>
    <td>with hardware Pre-processing</td>
  </tr>
  <tr>
    <td>tiny_yolo_v3</td>
    <td>39.91</td>
    <td>50.5</td>
    <td>26.5 %</td>
  </tr>
</table>
