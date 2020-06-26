# Apache TVM with Vitis-AI

Apache TVM is a versatile framework that integrates the high-performance computing power of Xilinx Vitis-AI DPUs with the flexibility of the TVM framework to accelerate models from many different training frameworks supported by TVM in a matter that is seamless to the end-user. 

### Background 

Apache TVM with Vitis-AI uses a number of projects as follows: 
* [Apache TVM] - An end-to-end deep learning compiler stack
* [Xilinx Vitis AI] - Xilinx development platform for AI inference
* [DPU] :  Xilinx Deep Learning Processor Unit (DPU)
* [Pynq-DPU] - DPU overlay for Pynq to run models compiled using TVM-Vitis flow on edge devices


### Installation
Apache TVM with Vitis-AI is provided through a docker image. The provided scripts and Dockerfile compiles TVM and Vitis-AI into a single image. 

The following command will create the TVM with Vitis-AI image

```sh
$ bash ./build.sh ci_vai_1x bash
```
This downloads Apache TVM, installs the necessary dependencies, and builds with Vitis-AI support.

### Running TVM

Once finished builiding the image, run the docker image using the run script.
```sh
$ bash ./bash.sh tvm.ci_vai_1x
# ...
# Now inside docker...
$ conda activate vitis-ai-tensorflow
```

The installation may be verified inside the docker image by importing the following packages in python3
```sh
$ python3
$ import tvm
$ import pyxir
```

While inside the docker, example of Apache TVM usage for different frameworks is provided in the "/opt/tvm-vai/tvm/tutorials/frontend" directories. Similarly, Vitis-AI examples using TVM are provided in the "/opt/tvm-vai/tvm/tutorials/accelerators" directory.

For more information on how to compile and run the Vitis-AI tutorials, refer to the "docs" directory in this folder.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job.)

   [Apache TVM]: https://tvm.apache.org/
   [Xilinx Vitis AI]: https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html
   [DPU]: https://www.xilinx.com/products/intellectual-property/dpu.html
   [Pynq-DPU]: https://github.com/Xilinx/DPU-PYNQ 
   [ZCU104]: https://www.xilinx.com/products/boards-and-kits/zcu104.html
  
  
