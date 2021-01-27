## Tensorflow SSD-Mobilenet Model
:pushpin: **Note:** This application can be run only on Alveo-U280 platform.

## Table of Contents

- [Introduction](#Introduction)
- [Set Up the target platform](#Setup)
- [Running the Application](#Running-the-Application)
- [Performance](#Performance)

## Introduction
The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. Accelerated post-processing(Sort and NMS) for ssd-mobilenet is provided and can only run on U280 board. In this application, software pre-process is used for loading input image, resize and mean subtraction.

## Setup
- Load and run the docker container.
```sh
./docker_run.sh -X xilinx/vitis-ai-cpu:<x.y.z>
# Activate Conda Environment
conda activate vitis-ai-caffe
```

### Data Preparation
- Download and extract coco datatset. (wget http://images.cocodataset.org/zips/val2017.zip)
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

### Download xclbin
- Download and extract xclbin tar.
- `wget -O waa_system_u280_v1.3.0.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=waa_system_u280_v1.3.0.tar.gz`
- `tar -xf waa_system_u280_v1.3.0.tar.gz && sudo cp dpu_ssdpost_u280.xclbin /usr/lib/dpu.xclbin`


## Build the Application
- `cd ${VAI_HOME}/demo/Whole-App-Acceleration/ssd_mobilenet/`
- `make build && make -j`

## Running the Application using Hardware accelerated post process
- `./run.sh model_ssd_mobilenet/ssd_mobilenet_v1_coco_tf.prototxt model_ssd_mobilenet/ssd_mobilenet_v1_coco_tf.xmodel <image directory> 1`

## Running the Application using Software post process
- `./run.sh model_ssd_mobilenet/ssd_mobilenet_v1_coco_tf.prototxt model_ssd_mobilenet/ssd_mobilenet_v1_coco_tf.xmodel <image directory> 0`

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
Pre-Process Execution time: 3393 us

DPU Execution time: 1265 us

FPGA Accelerated Post-Processing time: 200 us
