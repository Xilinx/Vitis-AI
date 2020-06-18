# Running on Zynq

## Overview


The TVM-Vitis flow occurs in two stages: Compilation and Execution. During the Compilation a user can choose to compile a model for the target devices that are currently supported by the flow. Subsequently, the generated files can be used to run models on a target device in the Execution. This document provides instruction to excute compiled models using the TVM-Vitis flow. For more information on how to compile models please refer to the "getting_start" document. 


The Xilinx Deep Learning Processor Unit ([DPU]) is a configurable computation engine dedicated for convolutional neural networks. On edge devices, The TVM-Vitis flow exploits DPU hardware accelerator built for the following evaluation boards:
* [Ultra96]
* [ZCU104]
* [ZCU102]

## Resources
You could find more information here:
*

## Board Setup
Follow instruction in [Pynq-DPU] repository to download and install DPU on one of the supported evaluation board.

### Test DPU on Pynq
You could try testing your DPU setup on your evaluation board.


```sh
$ dexplorer -w
```

If installed properly, the output should provide information on the DPU version installed on the board

### Setup TVM Runtime

The TVM-Vitis flow requires the TVM runtime to be installed on the board. Clone the tvm-release repository and run the setup script, as follows:

```sh
$ cd tvm-release/
$ bash zynq_setup.sh
```

This script clones the latest compatible TVM repository and builds the TVM runtime. 


## Executing a Compiled Model

Prior to running a model on the board, you need to compile the model for your targeted evlauation board and trasfer the compiled model on to the board. Please refer to the "getting_start.md" guide for compiling a model using the TVM-Vitis flow. 

The "tvm/tutorials/accelerators/run/" directory provides examples to run the example models in the "tvm/tutorials/accelerators/compile" directory.

Once you transfer the output directory from the compilation on to the board, you could use the provided scripts to run the model on the board. Below we present an example of running the mxnet_resnet_18 model using the mxnet_resnet_18.py script.


```sh
$ cd ${TVM_HOME}/tutorials/accelerators/run/
$ python3 -f "PATH_TO_COMPILED_MODEL"/mxnet_resnet_18/ -d "PATH_TO_COMPILED_MODEL"/mxnet_resnet_18/libdpu 
```

This script runs the model mxnet_resnet_18 model compiled using the TVM-Vitis flow on an image and produce the classification result.




[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. )

   [Ultra96]:  https://www.xilinx.com/products/boards-and-kits/1-vad4rl.html
   [ZCU104]: https://www.xilinx.com/products/boards-and-kits/zcu104.html
   [DPU]: https://www.xilinx.com/products/intellectual-property/dpu.html
   [Pynq-DPU]: https://github.com/Xilinx/DPU-PYNQ 
   [ZCU102]:  https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html
  
  
