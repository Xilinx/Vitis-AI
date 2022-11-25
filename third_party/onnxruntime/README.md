<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# ONNX Runtime with Xilinx Vitis AI DPU acceleration

Microsoft ONNX Runtime is a framework designed for high performance execution of ONNX models on a variety of platforms.

ONNX Runtime is enabled with Vitis AI and available through the Microsoft [ONNX Runtime](https://github.com/microsoft/onnxruntime) github page.

The current Vitis AI execution provider inside ONNX Runtime enables acceleration of Neural Network model inference with following Deep Learning Processor Units (DPU's):

#### DPU Targets
| Target Board  | DPU ID                           | ONNX Runtime Target ID                      |
|:-:|:-:|:-:|
| [U200]        | DPUCADF8H                        | DPUCADF8H                                   |
| [U250]        | DPUCADF8H                        | DPUCADF8H                                   |
| [U50]         | DPUCAHX8H <br /> DPUCAHX8H-DWC   | DPUCAHX8H-u50lv <br /> DPUCAHX8H-u50lv_dwc  |
| [U55C]        | DPUCAHX8H-DWC                    | DPUCAHX8H-u55c_dwc                          |
| [VCK5000]     | DPUCVDX8H <br /> DPUCVDX8H-DWC   | DPUCVDX8H <br /> DPUCVDX8H-dwc              |

In this directory you will find information on how to build ONNX Runtime with Vitis AI and on how to get started with an example.

## System Requirements
The [System Requirements](../../docs/reference/system_requirements.md) page lists system requirements for running docker containers as well as Alveo cards.

## Setup

This section provide the instructions for setting up the ONNX Runtime with Vitis AI flow for both cloud and edge. ONNX Runtime with Vitis AI support is provided through a docker container.

The following command will create the ONNX Runtime with Vitis AI docker image on the host machine

```sh
$ docker build -t onnxruntime-vitisai -f Dockerfile.vitis_ai_onnxrt .
```

Once finished builiding the container, run the docker image using the run script.
```sh
$ cd ../../
$ ./docker_run.sh onnxruntime-vitisai
# ...
# Now inside docker...
$ conda activate vitis-ai-tensorflow
```
The installation may be verified inside the docker image by importing the following packages in python3. Be sure to import pyxir before importing the `onnxruntime` package.

```sh
$ python3 -c "import pyxir; import onnxruntime"
```

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

## Getting started with an example

Inside the ONNX Runtime - Vitis AI docker, you can follow the instructions underneath to run an image classification example


1. Download minimal ImageNet validation dataset (step specific to this example):
   ```
   python3 -m ck pull repo:ck-env
   python3 -m ck install package:imagenet-2012-val-min
   ```
2. (Optional) set the number of inputs to be used for on-the-fly quantization to a lower number (e.g. 8) to decrease the quantization time (potentially at the cost of lower accuracy):
   ```
   export PX_QUANT_SIZE=8
   ```
3. Run the ResNet 18 example script:
   ```
   cd /workspace/external/onnxruntime
   python3 image_classification.py [DPU TARGET ID]
   ```
   Where the DPU target identifier can be found in the [DPU Targets](#dpu-targets) table above.
   After the model has been quantized and compiled using the first N inputs you should see accelerated execution of the 'images/dog.jpg' image with the DPU accelerator.


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job.)

   [ONNX Runtime - Vitis AI ExecutionProvider]: https://www.onnxruntime.ai/docs/reference/execution-providers/Vitis-AI-ExecutionProvider.html
   [U200]: https://www.xilinx.com/products/boards-and-kits/alveo/u200.html
   [U250]: https://www.xilinx.com/products/boards-and-kits/alveo/u250.html
   [U50]: https://www.xilinx.com/products/boards-and-kits/alveo/u50.html
   [U55C]: https://www.xilinx.com/products/boards-and-kits/alveo/u55c.html
   [VCK5000]: https://www.xilinx.com/products/boards-and-kits/vck5000.html
   [System Requirements]: ../../docs/learn/system_requirements.md
   [Alveo Setup]: ../../setup/alveo/README.md
   [VCK5000 Setup]: ../../setup/vck5000/README.md
