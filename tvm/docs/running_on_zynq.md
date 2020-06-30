# Running on Zynq

## Overview

 The TVM with Vitis AI flow contains two stages: Compilation and Execution. During the compilation a user can choose to compile a model for any of the target devices that are currently supported. Once a model is compiled, the generated files can be used to run the model on a target device during the Execution stage. Currently, the TVM with Vitis AI flow supports a selected number of Xilinx data center and edge devices.
 
This document provides instruction to execute compiled models using the TVM with Vitis AI flow on supported Zynq devices. For more information on how to compile models please refer to the "compiling_a_model.md" document. 


The Xilinx Deep Learning Processor Unit (DPU) is a configurable computation engine dedicated for convolutional neural networks. On edge devices, The TVM with Vitis AI flow exploits [DPUCZDX8G] hardware accelerator built for the following evaluation boards:
* [Ultra96]
* [ZCU104]
* [ZCU102]

## Resources
You could find more information here:
* Board setup - Follow instruction in [Pynq-DPU] repository to download and install DPUCZDX8G on one of the supported Zynq evaluation board.


### Test DPU on Pynq
You could try testing your DPU setup on your evaluation board.


```sh
# need sudo to access dpu drivers
$ sudo dexplorer -w
```

If installed properly, the output should provide information on the DPU version installed on the board

### Setup TVM Runtime

The TVM with Vitis AI flow requires the TVM runtime to be installed on the board. Clone the tvm-release repository and run the setup script, as follows:

```sh
$ cd tvm-release/
$ bash zynq_setup.sh
```

This script clones the latest compatible TVM repository and builds the TVM runtime. 


## Executing a Compiled Model

Prior to running a model on the board, you need to compile the model for your target evaluation board and transfer the compiled model on to the board. Please refer to the "compiling_a_model.md" guide for compiling a model using the TVM with Vitis AI flow. 

While inside the docker, the "/opt/tvm-vai/tvm/tutorials/accelerators/run" directory provides examples to run the example models in the "opt/tvm-vai/tvm/tutorials/accelerators/compile" directory.

Once you transfer the output directory from the compilation on to the board, you could use the provided scripts to run the model on the board. Below we present an example of running the mxnet_resnet_18 model using the mxnet_resnet_18.py script.


```sh
$ cd ${TVM_HOME}/tutorials/accelerators/run/
# need sudo to access dpu drivers
$ sudo python3 mxnet_resent_18.py -f "PATH_TO_COMPILED_MODEL"/mxnet_resnet_18/ -d "PATH_TO_COMPILED_MODEL"/mxnet_resnet_18/libdpu 
```

This script runs the model mxnet_resnet_18 model compiled using the TVM with Vitis AI flow on an image and produce the classification result.




[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. )

   [Ultra96]:  https://www.xilinx.com/products/boards-and-kits/1-vad4rl.html
   [ZCU104]: https://www.xilinx.com/products/boards-and-kits/zcu104.html
   [DPUCZDX8G]: https://www.xilinx.com/products/intellectual-property/dpu.html
   [Pynq-DPU]: https://github.com/Xilinx/DPU-PYNQ 
   [ZCU102]:  https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html
  
  
