# Running on Alveo

## Overview

 The TVM with Vitis AI flow contains two stages: Compilation and Execution. During the compilation a user can choose to compile a model for the target devices that are currently supported.
 
This document provides instruction to execute compiled models using the TVM with Vitis AI flow on supported Alveo devices. For more information on how to compile models please refer to the "compiling_a_model.md" document. 


## Resources
You could find more information here:
* Board Setup - Follow instruction in [Alveo Setup] repository to download and install dpuv1 on one of the supported Alveo board.


## Executing a Compiled Model

Prior to running a model on the board, you need to compile the model for your targeted evaluation board and transfer the compiled model on to the board. 

The "tvm/tutorials/accelerators/run/" directory provides examples to run the example models in the "tvm/tutorials/accelerators/compile" directory.

Below we present an example of running the mxnet_resnet_18 model using the mxnet_resnet_18.py script.


```sh
$ # In docker
$ conda activate vitis-ai-tensorflow
$ cd ${TVM_HOME}/tutorials/accelerators/run/
$ python3 -f "PATH_TO_COMPILED_MODEL"/mxnet_resnet_18/ -d "PATH_TO_COMPILED_MODEL"/mxnet_resnet_18/libdpu 
```

This script runs the model mxnet_resnet_18 model compiled using the TVM with Vitis AI flow on an image and produce the classification result.




[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. )

   [Alveo Setup]: https://github.com/Xilinx/Vitis-AI/tree/master/alveo
   [DPUv1]: https://github.com/Xilinx/Vitis-AI/blob/master/alveo/docs/ml-suite-overview.md  
