<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# Apache TVM with Vitis AI

Apache TVM is a versatile framework that integrates the high-performance computing power of Xilinx Vitis-AI DPUs with the flexibility of the TVM framework to accelerate models from many different training frameworks supported by TVM in a matter that is seamless to the end-user. 

The current Vitis AI flow inside TVM enables acceleration of Neural Network model inference on edge and cloud. The identifiers for the supported edge and cloud Deep Learning Processor Units (DPU's) are:

#### DPU Targets
| Target Board  | DPU ID                           | TVM Target ID                               |
|:-:|:-:|:-:|
| [U200]        | DPUCADF8H                        | DPUCADF8H                                   |
| [U250]        | DPUCADF8H                        | DPUCADF8H                                   |
| [U50]         | DPUCAHX8H <br /> DPUCAHX8H-DWC   | DPUCAHX8H-u50lv <br /> DPUCAHX8H-u50lv_dwc  |
| [U55C]        | DPUCAHX8H-DWC                    | DPUCAHX8H-u55c_dwc                          |
| [VCK5000]     | DPUCVDX8H <br /> DPUCVDX8H-DWC   | DPUCVDX8H <br /> DPUCVDX8H-dwc              |
| [VCK190]      | DPUCVDX8G                        | DPUCVDX8G                                   |
| [ZCU104]      | DPUCZDX8G                        | DPUCZDX8G-zcu104                            |
| [ZCU102]      | DPUCZDX8G                        | DPUCZDX8G-zcu102                            |
| [Kria SOM]    | DPUCZDX8G                        | DPUCZDX8G-kv260                             |
| [Ultra96]     | DPUCZDX8G                        | DPUCZDX8G-ultra96                           |

In this directory you will find information on how to build TVM with Vitis AI and on how to get started with an example.


## System Requirements
The [System Requirements] page lists system requirements for running docker containers as well as Alveo cards.
Deploying models on the edge requires a host machine for compiling models using the TVM with Vitis AI flow, and an edge device for running the compiled models. The host system requirements are the same as above.

## Setup

This section provide the instructions for setting up the TVM with Vitis AI flow for both cloud and edge. Apache TVM with Vitis AI support is provided through a docker container. The provided scripts and Dockerfile compiles TVM and Vitis AI into a single image.

The following command will create the TVM with Vitis AI image on the host machine

```sh
$ ./build.sh vitis_ai bash
```
This command downloads the latest Apache TVM repository, installs the necessary dependencies, and builds it with Vitis-AI support.

Once finished building the container, run the docker image using the run script.
```sh
$ cd ../../
$ ./docker_run.sh tvm.vitis_ai
# ...
# Now inside docker...
$ conda activate vitis-ai-tensorflow
```
The installation may be verified inside the docker image by importing the following packages in python3. Be sure to import pyxir before importing the TVM package.

```sh
$ python3 -c "import pyxir; import tvm"
```
The provided docker image can be used to compile models for the cloud and for the edge targets.


### Alveo Setup

Check out following page for setup information: [Alveo Setup].

**DPU IP Selection**

```
conda activate vitis-ai-tensorflow
# For Alveo DPU's
source /workspace/setup/alveo/setup.sh [DPU-IDENTIFIER]
```

The DPU identifier for this can be found in the second column of the [DPU Targets](#dpu-targets) table.

### Versal VCK5000 Setup

Check out following page for setup information: [VCK5000 Setup].

**DPU IP Selection**

```
conda activate vitis-ai-tensorflow
# For Versal DPU's 
source /workspace/setup/vck5000/setup.sh [DPU-IDENTIFIER]
```

The DPU identifier for this can be found in the second column of the [DPU Targets](#dpu-targets) table.

### Zynq Setup

For the Zynq target (DPUCZDX8G) the compilation stage will run inside the docker on a host machine. This doesn't require any specific setup except for building the docker.
For executing the model, the Zynq target will have to be set up and more information on that can be found in the [Running on Zynq] documentation.


## Next steps

* [Compiling a model](./docs/compiling_a_model.md)
* [Running on Alveo](./docs/running_on_alveo.md)
* [Running on Zynq]

For more details on how to compile and run models using TVM with the Vitis AI, you could refer to [Vitis AI Integration].

## Getting better throughput using AKS
TVM can run models efficiently but it does not support parallel execution of the whole pipeline. So, to get the maximum throughput for the whole application, we make use of the Vitis-AI pipelining framework [AKS (AI Kernel Scheduler)](../../tools/AKS/README.md#Introduction) at the application level. 

Follow [this link](./examples/AKS/README.md) to run TVM compiled graphs using AKS.

## Additional Resources

Apache TVM with Vitis-AI uses a number of projects as follows: 
* [Apache TVM]: An end-to-end deep learning compiler stack
* [Xilinx Vitis AI]: Xilinx development platform for AI inference
* [DPU] :  Xilinx Deep Learning Processor Unit (DPU)



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job.)

   [Apache TVM]: https://tvm.apache.org/
   [Xilinx Vitis AI]: https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html
   [DPU]: https://www.xilinx.com/products/intellectual-property/dpu.html
   [System Requirements]: ../../docs/learn/system_requirements.md
   [Pynq-DPU]: https://github.com/Xilinx/DPU-PYNQ 
   [ZCU104]: https://www.xilinx.com/products/boards-and-kits/zcu104.html
   [Ultra96]: https://www.xilinx.com/products/boards-and-kits/1-vad4rl.html
   [ZCU102]: https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html
   [Kria SOM]: https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html
   [Zynq Setup]: ./docs/docs/running_on_zynq.md#zynq-setup
   [Running on Zynq]: ./docs/running_on_zynq.md
   [Alveo]: https://www.xilinx.com/products/boards-and-kits/alveo.html
   [Alveo Setup]: ../../setup/alveo/README.md
   [VCK5000 Setup]: ../../setup/vck5000/README.md
   [U200]: https://www.xilinx.com/products/boards-and-kits/alveo/u200.html
   [U250]: https://www.xilinx.com/products/boards-and-kits/alveo/u250.html
   [U50]: https://www.xilinx.com/products/boards-and-kits/alveo/u50.html
   [U55C]: https://www.xilinx.com/products/boards-and-kits/alveo/u55c.html
   [VCK5000]: https://www.xilinx.com/products/boards-and-kits/vck5000.html
   [VCK190]: https://www.xilinx.com/products/boards-and-kits/vck190.html
   [Vitis AI Integration]: https://github.com/apache/tvm/blob/main/docs/how_to/deploy/vitis_ai.rst
   [Zynq Ultrascale+ MPSoc]: https://www.xilinx.com/products/silicon-devices/soc/zynq-ultrascale-mpsoc.html
