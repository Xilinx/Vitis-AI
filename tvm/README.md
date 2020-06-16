# Apache TVM with Vitis-AI

Apache TVM is a versatile framework that integrates the high-performance computing power of Xilinx Vitis-AI DPUs with the flexibility of the TVM framework to accelerate models from many different training frameworks supported by TVM in a matter that is seamless to the end-user. 

### Tech

Apache TVM with Vitis-AI uses a number of projects as follows: 
* [Apache TVM] - An end-to-end deep learning compiler stack
* [Xilinx Vitis AI] - Xilinx development platform for AI inference
* [DPU] :  Xilinx Deep Learning Processor Unit (DPU)
* [Pynq-DPU] - DPU overlay for Pynq to run models compiled using TVM-Vitis flow on edge devices


### Installation
Apache TVM with Vitis-AI is provided through a docker image. The provided scripts and Dockerfile build compile TVM and Vitis-AI into a single image. 

The following command will create the TVM with Vitis-AI image

```sh
$ bash ./build.sh ci_vai_11 bash
```
This download Apache TVM, install the necessary dependencies, and build with Vitis-AI support.

### Running TVM

Once finished builiding the image, run the docker image using the run script.
```sh
$ bash ./bash.sh tvm.ci_vai_11
# Now inside docker...
$ conda activate vitis-ai-tensorflow
```

The installation may be verified inside the docker image by importing the TVM-VAI packages in python3
```sh
$ python3
$ import tvm
$ import pyxir
```

Examples of Apache TVM usage for different frameworks is provided in the tutorials/frontend directories.

Vitis-AI examples using TVM are provided in the tutorials/accelerators directory.


License
----


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Apache TVM]: https://tvm.apache.org/
   [Xilinx Vitis AI]: https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html
   [DPU]: https://www.xilinx.com/products/intellectual-property/dpu.html
   [Pynq-DPU]: https://github.com/Xilinx/DPU-PYNQ 
   [ZCU104]: https://www.xilinx.com/products/boards-and-kits/zcu104.html
  
  
