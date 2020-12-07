## Tensorflow SSD-Mobilenet Model
* This application can be run only on Alveo-U280 platform.

## Table of Contents

- [Introduction](#Introduction)
- [Set Up the target platform](#Setup)
- [Running the Application](#Running-the-Application)

## Introduction
The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. Accelerated post-processing(Sort and NMS) for ssd-mobilenet is provided and can only run on U280 board. In this application, software pre-process is used for loading input image, resize and mean subtraction.

## Setup
```sh
# Activate Conda Environment
conda activate vitis-ai-caffe
```

### Data Preparation
- Download coco2014 datatset (https://cocodataset.org/#download)
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

### Download xclbin
- Download and extract xclbin tar. 
- `wget -O waa_system_u280_v1.3.0.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=waa_system_u280_v1.3.0.tar.gz`
- `tar -xf waa_system_u280_v1.3.0.tar.gz && sudo cp dpu_ssdpost_u280.xclbin /usr/lib/dpu.xclbin`

### Running the Application
- `cd /workspace/demo/Whole-App-Acceleration/ssd_mobilenet/`
- `make build && make -j`
- `./run.sh model_ssd_mobilenet/ssd_mobilenet_v1_coco_tf.prototxt model_ssd_mobilenet/ssd_mobilenet_v1_coco_tf.xmodel <image path>`

