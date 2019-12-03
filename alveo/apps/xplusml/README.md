# X Plus ML: Accelerating ML Preprocessing

## Introduction

This application demonstrates how Xilinx® [Vitis Vision library](https://xilinx.github.io/Vitis_Libraries/vision/) functions (X) can be integrated with deep neural network (DNN) accelerator to achieve complete application acceleration. This application focuses on accelerating the pre-processing involved in inference of classification networks (Googlenet_v1 and resnet-50).

## Background

Input images are preprocessed  before being fed for inference of different deep neural networks. The pre-processing steps vary from network to network. For example, for classification networks like Googlenet_v1 and resnet-50 the input image is resized to 224 x 224 size and then channel-wise mean subtraction is performed before feeding the data to the DNN accelerator. 

[Vitis Vision library](https://xilinx.github.io/Vitis_Libraries/vision/) provides functions optimized for FPGA devices that are drop-in replacements for standard OpenCV library functions. This application demonstrates how Vitis Vision library functions can be used to accelerate pre-processing.

Currently, this application can only run on Alveo-U200 device. Two processes are created one for running pre-processing kernel and one for running the ML accelerator. The pre-processed data is transferred to the ML accelerator over a queue. A shared library (.so) file is created which contains the openCL calls to create a handle for the pre-processing kernel and the handle is used to run the kernel.

## Running the Application

1. `cd $VAI_ALVEO_ROOT/apps/xplusml`
2. Use `run.sh` file to run the application. Familiarize yourself with the script usage by `./run.sh -h`

### Examples:

1. To run image classification using Googlenet_v1
```sh
$ ./run.sh
```

2. To run image classification using resnet-50
```sh
$ ./run.sh -m resnet50
```

## Results

Below table presents the comparison of the pre-processing latency on FPGA and CPUs for Googlenet_v1.



|              | Intel(R) Xeon(R)Silver 4100 CPU @ 2.10GHz, 8 core | Intel(R) Core(TM)  i7-4770 CPU @ 3.40GHz, 4 core | FPGA  (Alveo-U200) | Speedup  (Xeon/i7) |
|--------------|---------------------------------------------------|--------------------------------------------------|--------------------|--------------------|
| Googlenet_v1 | 5.63 ms                                           | 59.9 ms                                          | 1.1 ms             | 5x/54x             |