# Running on Alveo

## Overview

The TVM with Vitis AI flow contains two stages: Compilation and Execution. During the compilation a user can choose to compile a model for any of the target devices that are currently supported. Once a model is compiled, the generated files can be used to run the model on a target device during the Execution stage. Currently, the TVM with Vitis AI flow supports a selected number of Xilinx data center and edge devices.

This document provides instruction to execute compiled models using the TVM with Vitis AI flow on supported Alveo devices. For more information on how to compile models please refer to the [Compiling a model]("./compiling_a_model.md") document.

## Alveo Setup

You can find the setup instructions [here](../README.md#alveo-setup)


## Executing a Compiled Model

The examples directory provides scripts for compiling and running models.

Once you transfer the compiled model, you could use the provided scripts to run the model on the board. If the TVM with Vitis AI support docker is setup on a machine that includes an Alveo board, the compiled model can be executed directly in the docker. Below we present an example of running the MXNet ResNet 18 model using the run_mxnet_resnet_18.py script.


```sh
# Inside docker
$ conda activate vitis-ai-tensorflow
# DPU_TARGET options: 'DPUCADF8H', 'DPUCAHX8H-u50', 'DPUCAHX8H-u280', 'DPUCAHX8L', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102', 'DPUCZDX8G-som', 'DPUCZDX8G-ultra96
$ cd /workspace/third_party/tvm/examples
$ python3 run_mxnet_resnet_18.py -f "PATH_TO_COMPILED_TVM_MODEL (.so)"
```

This script runs the model mxnet_resnet_18 model compiled using the TVM with Vitis AI flow on an image and produces the classification result.

Following table shows all possible script flags:

| Flag         | Description                                              | Default   |
| -------------|----------------------------------------------------------| ----------|
| -f           | Path to the exported TVM compiled model (tvm_dpu_cpu.so in the example)|           |
| --iterations | The number of iterations that the model will be executed | 1         |





[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. )

   [Compiling a model]: "./compiling_a_model.md"
   [Alveo Setup]: ../../setup/alveo/README.md
   [DPUCADF8H]: ../../../setup/alveo/u200_u250/README.md
