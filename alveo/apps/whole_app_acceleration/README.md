# Whole Application Acceleration: Accelerating ML Preprocessing

## Introduction

This application demonstrates how Xilinx® [Vitis Vision library](https://xilinx.github.io/Vitis_Libraries/vision/) functions can be integrated with deep neural network (DNN) accelerator to achieve complete application acceleration. This application focuses on accelerating the pre-processing involved in inference of classification networks (Googlenet_v1 and resnet-50) and object detection networks (yolo_v3 and tiny_yolo_v3).

## Background

Input images are preprocessed  before being fed for inference of different deep neural networks. The pre-processing steps vary from network to network. For example, for classification networks like Googlenet_v1 and resnet-50 the input image is resized to 224 x 224 size and then channel-wise mean subtraction is performed before feeding the data to the DNN accelerator. 

[Vitis Vision library](https://xilinx.github.io/Vitis_Libraries/vision/) provides functions optimized for FPGA devices that are drop-in replacements for standard OpenCV library functions. This application demonstrates how Vitis Vision library functions can be used to accelerate pre-processing.

Currently, applications accelerating pre-processing for classification networks (Googlenet_v1 and resnet-50) and YOLO are provided and  can only run on Alveo-U200 device (with xilinx_u200_xdma_201830_1 shell). For each application, there is an option to run with and without accelerating JPEG deocding. Two processes are created one for running pre-processing kernel and one for running the ML accelerator. The pre-processed data is transferred to the ML accelerator over a queue. 

