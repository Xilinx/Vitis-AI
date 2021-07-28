# Whole Application Acceleration: Accelerating ML Pre/Post-processing for Classification and Detection networks

## Introduction

These examples demonstrate how Xilinx® [Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) functions can be integrated with deep neural network (DNN) accelerator to achieve complete application acceleration. This application focuses on accelerating the pre/post-processing involved in inference of object detection & classification networks.

## Background

Input images are preprocessed  before being fed for inference of different deep neural networks. The pre-processing steps vary from network to network. For example, for classification networks like Resnet-50 the input image is resized to 224 x 224 size and then channel-wise mean subtraction is performed before feeding the data to the DNN accelerator. For detection networks like YOLO v3 the input image is resized to 256 x 512 size using letterbox before feeding the data to the DNN accelerator. Similarly, some networks like SSD-Mobilenet also require the inference output to be processed to filter out unncessary data. Non Maximum Supression (NMS) is an example of post-processing function.


[Vitis Vision library](https://github.com/Xilinx/Vitis_Libraries/tree/master/vision) provides functions optimized for FPGA devices that are drop-in replacements for standard OpenCV library functions. These application demonstrates how Vitis Vision library functions can be used to accelerate the complete application.

## Model Performance
Refer below table for examples with demonstration on complete acceleration of various applications where pre/post-processing of several networks are accelerated on different target platforms.


| No. | Application                                           | Backbone Network | Accelerated Part(s)   | H/W Accelerated Functions                     | DPU Supported (% Improvement Over Non-WAA App) |
|-----|-------------------------------------------------------|------------------|----------------------------|-----------------------------------------------|--------------------------------------------------------------------------|
| 1   | adas_detection                                        | Yolo v3          | Pre-process                | resize, letter box, scale                     | ZCU102  (*64%) , ALVEO-U50 (*44%)                                        |
| 2   | adas_detection_versal                                 | Yolo v3          | Pre-process                | resize, letter box, scale                     | VCK190 (*14.17%)                                                         |
| 3   | adas_detection_int8                                   | Yolo v3          | Pre-process                | resize, letter box, scale                     | ALVEO-U200 (**83%)                                                       |
| 4   | adas_detection_zero_copy                              | Yolo v3          | Pre-process                | resize, letter box, scale                     | ZCU102  (*89%)                                                           |
| 5   | fall_detection                                        | VGG16            | Pre-process                | Lucas-Kanade Dense Non-Pyramidal Optical Flow | ALVEO-U200 (*6%)                                                        |
| 6   | resnet50_int8                                         | resnet-50        | Pre-process                | resize, mean subtraction, scale               | ALVEO-U200 (37%)                                                         |
| 7   | resnet50_jpeg                                         | resnet-50        | Pre-process                | JPEG decoder, resize, mean subtraction, scale | ZCU102  (*69%)                                                           |
| 8   | resnet50_mt_py                                        | resnet-50        | Pre-process                | resize, mean subtraction, scale               | ZCU102 (*61%), ALVEO-U50 (*22%)                                          |
| 9   | resnet50_versal                                       | resnet-50        | Pre-process                | resize, mean subtraction                      | VCK190 (44.5%)                                                           |
| 10  | resnet50_zero_copy                                    | resnet-50        | Pre-process                | resize, mean subtraction, scale               | ZCU102 (29%)                                                             |
| 11  | ssd_mobilenet_zero_copy                               | mobilenet        | pre-process & post-process | resize, BGR2RGB, scale, sort, NMS             | ALVEO-U280 (74%)                                                         |
| 12  | SORT |                  | Post-process               | kalman filters                                | ZCU102 (**59%)                                                           |


`*` Includes imread/jpeg-decoder latency in both WAA and non WAA app

`**` Impact shown is on pre/post processing acceleration

:pushpin: **Note:** Non-WAA applications use equivalent OpenCV functions. As OpenCV's LK OF is sparse, equivalent non-WAA application uses OpenCV's Farneback OF algorithm with 10 threads.
