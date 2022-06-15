# Running on Zynq

## Overview

 The TVM with Vitis AI flow contains two stages: Compilation and Execution. During the compilation stage a user can choose to compile a model for any of the target devices that are currently supported. Once a model is compiled, the generated files can be used to run the model on a target device during the Execution stage. Currently, the TVM with Vitis AI flow supports a selected number of Xilinx data center and edge devices.
 
This document provides instruction to execute compiled models using the TVM with Vitis AI flow on supported Zynq devices. For more information on how to compile models please refer to the [Compiling a model](./compiling_a_model.md) document. 


The Xilinx Deep Learning Processor Unit (DPU) is a configurable computation engine dedicated for convolutional neural networks. On edge devices, The TVM with Vitis AI flow exploits [DPUCZDX8G] hardware accelerator built for the following evaluation boards:

* [ZCU104]
* [ZCU102]
* [Ultra96]
* [Kria SOM]

## Zynq Setup
You could find more information on how to setup your target device below:

### Petalinux setup
1. Download the Petalinux image for your target:
    * [ZCU102](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu102-dpu-v2022.1-v2.5.0.img.gz)
    * [ZCU104](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu104-dpu-v2022.1-v2.5.0.img.gz)
    * [KV260](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-kv260-dpu-v2022.1-v2.5.0.img.gz)
2. Use Etcher software to burn the image file onto the SD card.
3. Insert the SD card with the image into the destination board.
4. Plug in the power and boot the board using the serial port to operate on the system.
5. Set up the IP information of the board using the serial port.
6. Install TVM and the TVM - Vitis AI flow dependencies. This can be done using the [petalinux setup script](../petalinux_setup.sh) and will take around 1 hour. Note that this script will create 4GB of swap space so make sure that your SD card has enough space.

```sh
bash petalinux_setup.sh
```


**For details on steps 2-5, please refer to [Setting Up the Evaluation Board](https://www.xilinx.com/html_docs/vitis_ai/1_3/installation.html#yjf1570690235238)**


### Test TVM installation
You can test your setup with:


```sh
$ python3 -c "import pyxir; import tvm"
```

## Executing a Compiled Model

Prior to running a model on the board, you need to compile the model for your target evaluation board and transfer the compiled model on to the board. Please refer to the [Compiling a model](./compiling_a_model.md) guide for compiling a model using the TVM - Vitis AI flow. 

The examples directory includes script to compile and run the model. Once you transfer the compiled model to the device, you can use the provided script from the examples directory to run the model. Below we provide the command for running the MXNet ResNet 18 model using the "run_mxnet_resnet_18.py" script.


```sh
$ python3 run_mxnet_resnet_18.py -f "PATH_TO_COMPILED_TVM_MODEL (.so)"
```

Following table shows all possible script flags:

| Flag         | Description                                              | Default   |
| -------------|----------------------------------------------------------| ----------|
| -f           | Path to the exported TVM compiled model (tvm_dpu_cpu.so in the example)    |           |
| --iterations | The number of iterations that the model will be executed | 1         |


Additionally, you can use the "run_mxnet_resnet_18_zynq_fps.py" script to return the number of frames per second (FPS) achieved:

```sh
python3 run_mxnet_resnet_18_zynq_fps.py -f "PATH_TO_COMPILED_TVM_MODEL (.so)"
```

Following table shows all possible script flags:

| Flag                 | Description                                              | Default   |
| ---------------------|----------------------------------------------------------| ----------|
| -f                   | Path to the exported TVM compiled model (tvm_dpu_cpu.so in the example)    |           |
| -t                   | Number of threads to use. Set larger than one to use multiple DPU Compute Units (CU's) in parallel if possible. | 1         |
| --iterations         | The number of iterations that the model will be executed. Increase this number to calculate frames per second (FPS) over a larger number of runs. | 1000         |
| --nb_tvm_threads     | The number of CPU threads being used by TVM. You will have to limit this number to get best CPU - DPU performance in multi-threaded execution. Dedault None means that TVM will use as many threads as possible. | None         |

For example, to use multiple DPU Compute Units (CU's) in parallel, increase the number of threads and limit the number of TVM CPU threads:
```sh
python3 run_mxnet_resnet_18_zynq_fps.py -f "PATH_TO_COMPILED_TVM_MODEL (.so)" -t 3 --nb_tvm_threads 1
```

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. )

   [Ultra96]:  https://www.xilinx.com/products/boards-and-kits/1-vad4rl.html
   [ZCU104]: https://www.xilinx.com/products/boards-and-kits/zcu104.html
   [DPUCZDX8G]: https://www.xilinx.com/products/intellectual-property/dpu.html
   [Pynq-DPU]: https://github.com/Xilinx/DPU-PYNQ 
   [Vitis-AI User Guide]: https://www.xilinx.com/cgi-bin/docs/rdoc?t=vitis_ai;v=latest;d=zkj1576857115470.html
   [ZCU102]:  https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html
   [Kria SOM]: https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html
  
  
