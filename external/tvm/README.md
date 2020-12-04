# Apache TVM with Vitis-AI

Apache TVM is a versatile framework that integrates the high-performance computing power of Xilinx Vitis-AI DPUs with the flexibility of the TVM framework to accelerate models from many different training frameworks supported by TVM in a matter that is seamless to the end-user. 

The current Vitis-AI Byoc flow inside TVM enables acceleration of Neural Network model inference on edge and cloud. The identifiers for the supported edge and cloud Deep Learning Processor Units (DPU's) are DPUCZDX8G respectively DPUCADX8G. DPUCZDX8G and DPUCADX8G are hardware accelerators for convolutional neural networks (CNN's) deployed on the Xilinx [Zyq Ultrascale+ MPSoc] [Alveo] (U200/U250) platforms, respectively.

In this repository you will find information on how to build TVM with Vitis-AI and on how to get started with an example.

## Background 

Apache TVM with Vitis-AI uses a number of projects as follows: 
* [Apache TVM] - An end-to-end deep learning compiler stack
* [Xilinx Vitis AI] - Xilinx development platform for AI inference
* [DPU] :  Xilinx Deep Learning Processor Unit (DPU)
* [Pynq-DPU] - DPU overlay for Pynq to run models compiled using TVM-Vitis flow on edge devices





## System Requirements
#### Cloud(DPUCADX8G)
The following table list the requirement for compiling and running models using the TVM with Vitis AI flow on Cloud:
| Component  | Requirement  |
|:-:|:-:|
|  Motherboard | PCI Express 3.0-compliant with one dual-width x16 slot  |
|  System Power Supply | 225W  |
| Operating System  | Ubuntu 16.04, 18.04  |
|   | CentOS 7.4, 7.5  |
|   | RHEL 7.4, 7.5  |
| CPU  | Intel i3/i5/i7/i9/Xeon 64-bit CPU   |
| GPU (Optional to accelerate quantization)  |  NVIDIA GPU with a compute capability > 3.0 |
| CUDA Driver (Optional to accelerate quantization)  | nvidia-410  |
| FPGA  | Xilinx Alveo U200 or U250  |
| Docker Version  |  19.03.1 |
#### Edge(DPUCZDX8G)
Deploying models on the edge requires a host machine for compiling models using the TVM with Vitis AI flow, and an edge device for running the compiled models. 

##### Host Requirements 
The following table presents the host machine requirements:
| Component  | Requirement  |
|:-:|:-:|
| Operating System  |  Ubuntu 16.04, 18.04 |
|   | CentOS 7.4, 7.5  |
|   | RHEL 7.4, 7.5  |
| CPU  | Intel i3/i5/i7/i9/Xeon 64-bit CPU  |
| GPU (Optional to accelerate quantization) | NVIDIA GPU with a compute capability > 3.0  |
| CUDA Driver  |  nvidia-410 |
| FPGA  |  Not necessary on host |
| Docker Version  |  19.03.1 |

##### Device Requirements
The TVM with Vitis AI flow currently supports the [Zyq Ultrascale+ MPSoc] devices speficied in the following table. The "TVM Identifier" is used during compilation stage to target a specific edge device.

| Target Board  | TVM Identifier|
|:-:|:-:|
| [Ulra96]  | DPUCZDX8G-ultra96 |
| [ZCU104]  | DPUCZDX8G-zcu104  |
| [ZCU102]  | DPUCZDX8G-zcu102  |


## Docker Build Instruction
This section provide the instructions for setting up the TVM with Vitis-AI flow for both cloud and edge. Apache TVM with Vitis-AI support is provided through a docker container. The provided scripts and Dockerfile compiles TVM and Vitis-AI into a single image.

The following command will create the TVM with Vitis-AI image on the host machine

```sh
$ bash ./build.sh ci_vai_1x bash
```
This command downloads the latest Apache TVM repository, installs the necessary dependencies, and builds it with Vitis-AI support.

## Docker Run Instruction

Once finished builiding the container, run the docker image using the run script.
```sh
$ bash ./bash.sh tvm.ci_vai_1x
# ...
# Now inside docker...
$ conda activate vitis-ai-tensorflow
```
The installation may be verified inside the docker image by importing the following packages in python3. Be sure to import pyxir before importing the TVM package.

```sh
$ python3
$ import pyxir
$ import tvm
```
The provided docker image can be used to compile models for the cloud and for the edge targets. 


Examples and documentations of compiling and running models using Apache TVM with Vitis AI support are provided in the "examples" and "docs" directories. Once inside the docker container, you could copy the examples directory from "/workspace/examples/" into the home directory and run the examples.

For more details on how to compile and run models using TVM with the Vitis-AI, you could refer to [Vitis-AI Integration].


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job.)

   [Apache TVM]: https://tvm.apache.org/
   [Xilinx Vitis AI]: https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html
   [DPU]: https://www.xilinx.com/products/intellectual-property/dpu.html
   [Pynq-DPU]: https://github.com/Xilinx/DPU-PYNQ 
   [ZCU104]: https://www.xilinx.com/products/boards-and-kits/zcu104.html
   [Ulra96]: https://www.xilinx.com/products/boards-and-kits/1-vad4rl.html
   [ZCU102]: https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html
   [Alveo]: https://www.xilinx.com/products/boards-and-kits/alveo.html
   [Alveo Setup]: https://github.com/Xilinx/Vitis-AI/tree/master/alveo
   [Vitis-AI Integration]: https://github.com/apache/incubator-tvm/blob/main/docs/deploy/vitis_ai.rst
   [Zyq Ultrascale+ MPSoc]: https://www.xilinx.com/products/silicon-devices/soc/zynq-ultrascale-mpsoc.html
