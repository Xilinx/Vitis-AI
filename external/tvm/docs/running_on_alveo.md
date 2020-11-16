# Running on Alveo

## Overview

 The TVM with Vitis AI flow contains two stages: Compilation and Execution. During the compilation a user can choose to compile a model for any of the target devices that are currently supported. Once a model is compiled, the generated files can be used to run the model on a target device during the Execution stage. Currently, the TVM with Vitis AI flow supports a selected number of Xilinx data center and edge devices.
 
This document provides instruction to execute compiled models using the TVM with Vitis AI flow on supported Alveo devices. For more information on how to compile models please refer to the "compiling_a_model.md" document. 


The Xilinx Deep Learning Processor Unit (DPU) is a configurable computation engine dedicated for convolutional neural networks. On data-center, The TVM with Vitis AI flow exploits [DPUCADX8G] hardware accelerator built for the Alveo device.

## Resources
You could find more information here:
* Board Setup - Follow instruction in [Alveo Setup] repository to download and install DPUCADX8G on one of the supported Alveo board.


## Executing a Compiled Model

Prior to running a model on the board, you need to compile the model for your targeted evaluation board and transfer the compiled model on to the board. Please refer to the "compiling_a_model.md" guide for compiling a model using the TVM with Vitis AI flow. 

While inside the docker the "/opt/tvm-vai/tvm/tutorials/accelerators/run" directory provides examples to run the example models in the "/opt/tvm-vai/tvm/tutorials/accelerators/compile" directory.

Once you transfer the output directory from the compilation on to the board, you could use the provided scripts to run the model on the board. If the TVM with Vitis AI support docker is setup on a machine that includes an Alveo board, the compiled model can be executed directly in the docker. Below we present an example of running the mxnet_resnet_18 model using the mxnet_resnet_18.py script. Ensure to source the proper XRT runtime path before running on the board.


```sh
# Inside docker
$ conda activate vitis-ai-tensorflow
$ cd ${TVM_HOME}/tutorials/accelerators/run/
$ python3 mxnet_resent_18.py -f /opt/tvm-vai/tvm/tutorials/accelerators/compile/mxnet_resnet_18/ -d /opt/tvm-vai/tvm/tutorials/accelerators/compile/mxnet_resnet_18/libdpu 
```

This script runs the model mxnet_resnet_18 model compiled using the TVM with Vitis AI flow on an image and produce the classification result.




[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. )

   [Alveo Setup]: https://github.com/Xilinx/Vitis-AI/tree/master/alveo
   [DPUCADX8G]: https://github.com/Xilinx/Vitis-AI/blob/master/alveo/docs/ml-suite-overview.md  
