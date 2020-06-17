# TVM-VAI

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The TVM-Vitis is a versatile flow that integrates the high-performance computing power of Xilinx’s vitis AI engine with the flexibility of the TVM framework to accelerate models from any given framework supported by the TVM in a matter that is seamless to the end-user. 

## Overview

The TVM-Vitis flow cointains two stages: Compilation and Execution. During the compilation a user can choose to compile a model for the target devices that are currently supported by the flow. Subsequently, the generated files can be used to run models on a target device. This document provide instructions to compile deep learning models using the TVM-Vitis flow. Currently, the TVM-Vitis flow supported a selected number of Xilinx data center and edge devices.

Prior to compiling models, you need to install setup TVM-Vitis flow, as described below.

## Installation
The TVM-Vitis provides an easy installation using Docker.
The repository provides scripts to build and run a docker image pre-setup for TVM-VAI. 

Simply clone the repositroy and use the build script to build an image. This step may take several moments.

```sh
$ cd tvm_release
$ bash ./build.sh ci_vai_11 bash
```
This creates an image based on Vitis-AI, pulls necessary dependencies, and builds the latest TVM-VAI compatible version of the Apache TVM.

## Development

Once finished builiding the image, run the docker image using the run script.
```sh
$ bash ./bash.sh tvm.ci_vai_11
```

### Environment Setup
TVM-VAI docker image uses conda package management system. Use conda to setup your docker environment at login prior to running any examples.

```sh
$ conda activate vitis-ai-tensorflow
```

Verify the setup by importing the TVM-VAI packages in python3
```sh
$ python3
$ import tvm
$ import pyxir
```
### Compilation

The compilation tutorials scripts in "tvm/tutorials/accelerators/compile/" directory demonstrate the Compiltion step using the TVM-Vitis flow. Run the tutorials once conda package "vitis-ai-tensorflow" is properly activated in the docker environment.

```sh
$ python3 mxnet_resnet_18.py
```

Each tutorial script generate a directory that includes the compiled files and libraries for running on the selected target device. This directory is required to run the model using the TVM-Vitis flow. For edge devices, the directory needs to be copied over to the target device.

#### Compiling for Alveo Board

The target device can be changed in the provided tutorial scripts. By default, the models are compiled for the "dpuv1" computation engine, targetting Alveo Board. 

#### Compiling for Zynq Board

You could also compile the example tutorials for supported Zynq devices. To change the compilation target, you could modify the "target" parameter in the provided tutorials to one of the supported platforms by the TVM-Vitis flow, such as "dpuv2-zcu104".

### Execution

The execution tutorials scripts in "tvm/tutorials/accelerators/run/" directory demonstrate running compiled models using the TVM-Vitis flow.

Please refer to "running_on_alveo.md" and "running_on_zynq.md" for instructions to run a compiled model on data center and edge devices, respectively.

License
----
MIT
