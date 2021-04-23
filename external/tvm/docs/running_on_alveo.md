# Running on Alveo

## Overview

 The TVM with Vitis AI flow contains two stages: Compilation and Execution. During the compilation a user can choose to compile a model for any of the target devices that are currently supported. Once a model is compiled, the generated files can be used to run the model on a target device during the Execution stage. Currently, the TVM with Vitis AI flow supports a selected number of Xilinx data center and edge devices.

This document provides instruction to execute compiled models using the TVM with Vitis AI flow on supported Alveo devices. For more information on how to compile models please refer to the "compiling_a_model.md" document.


The Xilinx Deep Learning Processor Unit (DPU) is a configurable computation engine dedicated for convolutional neural networks. The TVM with Vitis AI flow exploits [DPUCADX8G] hardware accelerator built for the Alveo device that is used for cloud applications.

## Resources
You could find more information here:
* Board Setup - Follow the instructions in [Alveo Setup] repository to download and install DPUCADX8G on one of the supported Alveo boards.


## Executing a Compiled Model

Prior to running a model on the board, you need to compile the model for your target evaluation board and transfer the compiled model (.so) on to the board. Please refer to the [Compiling a model](compiling_a_model.md) guide for compiling a model using the TVM with Vitis AI flow.

The examples directory provides scripts for compiling and running models.

Once you transfer the compile model, you could use the provided scripts to run the model on the board. If the TVM with Vitis AI support docker is setup on a machine that includes an Alveo board, the compiled model can be executed directly in the docker. Below we present an example of running the mxnet_resnet_18 model using the run_mxnet_resnet_18.py script.


```sh
# Inside docker
$ conda activate vitis-ai-tensorflow
$ python3 run_mxnet_resnet_18.py -f "PATH_TO_COMPILED_TVM_MODEL (.so)"
```

This script runs the model mxnet_resnet_18 model compiled using the TVM with Vitis AI flow on an image and produces the classification result.

Following table shows all possible script flags:

| Flag         | Description                                              | Default   |
| -------------|----------------------------------------------------------| ----------|
| -f           | Path to the exported TVM compiled model (tvm_dpu_cpu.so in the example)|           |
| --iterations | The number of iterations that the model will be executed | 1         |





[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. )

   [Alveo Setup]: ../../../setup/alveo/README.md
   [DPUCADX8G]: ../../../setup/alveo/u200_u250/README.md
