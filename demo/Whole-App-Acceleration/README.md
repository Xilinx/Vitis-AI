<<<<<<< HEAD
# Whole Application Acceleration: Accelerating ML Pre/Post-processing for Classification and Detection networks

## Introduction

These examples demonstrate how Xilinx® [Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) functions can be integrated with deep neural network (DNN) accelerator to achieve complete application acceleration. This application focuses on accelerating the pre/post-processing involved in inference of object detection & classification networks.

## Background

Input images are preprocessed  before being fed for inference of different deep neural networks. The pre-processing steps vary from network to network. For example, for classification networks like Resnet-50 the input image is resized to 224 x 224 size and then channel-wise mean subtraction is performed before feeding the data to the DNN accelerator. For detection networks like YOLO v3 the input image is resized to 256 x 512 size using letterbox before feeding the data to the DNN accelerator. Similarly, some networks like SSD-Mobilenet also require the inference output to be processed to filter out unncessary data. Non Maximum Supression (NMS) is an example of post-processing function.


[Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) provides functions optimized for FPGA devices that are drop-in replacements for standard OpenCV library functions. These application demonstrates how Vitis Vision library functions can be used to accelerate the complete application.

Refer below examples for demonstration of complete acceleration of various applications where pre/post-processing of several networks are accelerated on different target platforms.


## Classification network (Resnet-50) pre-process acceleration on ZCU102/Alveo-U50:

This example can be run on ZCU102 or Alveo-U50 platforms. Refer [README](./resnet50_mt_py_waa/README.md) for further details and steps to run.

## ADAS detection pre-process acceleration on ZCU102/Alveo-U50:

This example demonstrates acceleration of pre-processing of YOLO-v3 network. Refer [README](./adas_detection_waa/README.md) for further details and steps to run.

## Classification network (Resnet-50/Inception-v1) pre-process acceleration on Alveo-U200:

Refer [README](./classification/README.md) for further details and steps to run.

## Detection network pre-process acceleration on Alveo-U200:

This example demonstrates acceleration of pre-processing of tiny_yolo_v3 network on Alveo-U200. Refer [README](./yolo/README.md) for further details and steps to run.

## SSD-Mobilenet post-process acceleration on Alveo-U280:

This example demonstrates acceleration of post-processing of ssd-mobilenet network on Alveo-U280. Refer [README](./ssd_mobilenet/README.md) for further details and steps to run the example.






=======
# Whole Application Acceleration: Accelerating ML Pre/Post-processing for Classification and Detection networks

## Introduction

These examples demonstrate how Xilinx® [Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) functions can be integrated with deep neural network (DNN) accelerator to achieve complete application acceleration. This application focuses on accelerating the pre/post-processing involved in inference of object detection & classification networks.

## Background

Input images are preprocessed  before being fed for inference of different deep neural networks. The pre-processing steps vary from network to network. For example, for classification networks like Resnet-50 the input image is resized to 224 x 224 size and then channel-wise mean subtraction is performed before feeding the data to the DNN accelerator. For detection networks like YOLO v3 the input image is resized to 256 x 512 size using letterbox before feeding the data to the DNN accelerator. Similarly, some networks like SSD-Mobilenet also require the inference output to be processed to filter out unncessary data. Non Maximum Supression (NMS) is an example of post-processing function.


[Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) provides functions optimized for FPGA devices that are drop-in replacements for standard OpenCV library functions. These application demonstrates how Vitis Vision library functions can be used to accelerate the complete application.

Refer below examples for demonstration of complete acceleration of various applications where pre/post-processing of several networks are accelerated on different target platforms.


## Classification network (Resnet-50) pre-process acceleration on ZCU102/Alveo-U50:

This example can be run on ZCU102 or Alveo-U50 platforms. Refer [README](./resnet50_mt_py_waa/README.md) for further details and steps to run.

## ADAS detection pre-process acceleration on ZCU102/Alveo-U50:

This example demonstartes acceleration of pre-processing of YOLO-v3 network. Refer [README](./adas_detection_waa/README.md) for further details and steps to run.

## Classification network (Resnet-50/Inception-v1) pre-process acceleration on Alveo-U200:

Refer [README](./classification/README.md) for further details and steps to run.

## Detection network pre-process acceleration on Alveo-U200:

This example demonstartes acceleration of pre-processing of tiny_yolo_v3 network on Alveo-U200. Refer [README](./yolo/README.md) for further details and steps to run.

## SSD-Mobilenet post-process acceleration on Alveo-U280:
The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. Accelerated post-processing(Sort and NMS) for ssd-mobilenet is provided and can only run on U280 board. In this application, software pre-process is used for loading input image, resize and mean subtraction.

Refer [README](ssd_mobilenet/README.md) for further details on steps to run the example.
>>>>>>> 011958063f1df75324ddb8b126bfdbdaf632c8a1
