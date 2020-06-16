# TVM-VAI

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The TVM-Vitis is a versatile flow that integrates the high-performance computing power of Xilinx’s vitis AI engine with the flexibility of the TVM framework to accelerate models from any given framework supported by the TVM in a matter that is seamless to the end-user. 

### Tech

TVM-Vitis uses a number of projects as follows: 
* [Apache TVM] - An end-to-end deep learning compiler stack
* [Xilinx Vitis AI] - Xilinx development platform for AI inference
* [DPU] :  Xilinx Deep Learning Processor Unit (DPU)
* [Pynq-DPU] - DPU overlay for Pynq to run models compiled using TVM-Vitis flow on edge devices


### Installation
The TVM-Vitis provides an easy installation using Docker.
The repository provides scripts to build and run a docker image pre-setup for TVM-VAI. 

Simply clone the repositroy and use the build script to build an image. This step may take several moments.

```sh
$ cd tvm_release
$ bash ./build.sh ci_vai_11 bash
```
This creates an image based on Vitis-AI, pulls necessary dependencies, and builds the latest TVM-VAI compatible version of the Apache TVM.

### Development

Once finished builiding the image, run the docker image using the run script.
```sh
$ bash ./bash.sh tvm.ci_vai_11
```

##### Environment Setup
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

License
----


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Apache TVM]: https://tvm.apache.org/
   [Xilinx Vitis AI]: https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html
   [DPU]: https://www.xilinx.com/products/intellectual-property/dpu.html
   [Pynq-DPU]: https://github.com/Xilinx/DPU-PYNQ 
   [ZCU104]: https://www.xilinx.com/products/boards-and-kits/zcu104.html
  
  
