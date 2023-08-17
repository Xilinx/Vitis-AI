<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# Overview

WeGO (<u>W</u>hol<u>e</u> <u>G</u>raph <u>O</u>ptimizer) is a new feature of Vitis AI that aims to offer a smooth solution for deploying various models on cloud DPU. It achieves this by integrating the Vitis AI Development kit with TensorFlow 1.x, TensorFlow 2.x, and PyTorch frameworks.

WeGO automatically performs subgraph partitioning for models quantized by the Vitis AI quantizer. It applies optimizations and acceleration for the cloud DPU-compatible subgraphs. The remaining parts of the graph that are not supported by the DPU are dispatched to the framework for CPU execution. WeGO takes care of the whole graph optimization, compilation, and runtime subgraph dispatch and execution. This entire process is completely transparent to end users, making it very easy to use.

Using WeGO provides a straightforward transition from training to inference for model designers. WeGO offers a Python programming interface to deploy quantized models across different frameworks. This allows for maximum reuse of Python code, including pre-processing and post-processing, developed during the model training phase with frameworks. This greatly speeds up the deployment and evaluation of models over cloud DPUs.


# Preparation

## Setup Host Environment for Cloud
Before running the examples, please follow [setup for V70](https://xilinx.github.io/Vitis-AI/3.5/html/docs/quickstart/v70.html#alveo-v70-setup) for instructions on how to set up the host environment for V70 and make sure you have entered the Vitis-AI CPU docker container successfully and the DPU IP has been selected properly.

> Note:
>
> 1. VCK5000 PROD is deprecated since VAI 3.5. Please apply Vitis AI 3.0 for VCK5000 PROD usage.
>
> 2. Currently three different docker image targeting diverse AI frameworks are provided, make sure the right docker image is selected for the corresponding examples running purpose(i.e. TensorFlow 1.x docker for TensorFlow 1.x WeGO examples, TensorFlow 2.x docker for TensorFlow 2.x WeGO examples, PyTorch docker for PyTorch WeGO examples).

## Prepare WeGO Example Recipes
Download and prepare the WeGO examples recipes(ie. models and images) by executing:
```bash
$ wget -O wego_example_recipes.tar.gz http://www.xilinx.com/bin/public/openDownload?filename=wego_example_recipes.tar.gz
$ tar xf wego_example_recipes.tar.gz -C /tmp
```

# Run WeGO Examples

Please refer to the following links to run the wego demos targeting different frameworks according to corresponding instructions.

- [PyTorch](./pytorch) 
- [TensorFlow 2.x](./tensorflow-2.x)
- [TensorFlow 1.x](./tensorflow-1.x)

## QuickStart guide

The step-by-step instructions are made for easy start with the WeGO tool for model optimization.
Follow the guide to quickly launch the tool, for guide instructions, PyTorch InceptionV3 model was picked as an example 
to do [the compiling an offline quantized model](https://github.com/Xilinx/Vitis-AI/tree/master/examples/wego/pytorch/01_compiling_offline_quantized_models) and run it.
1. Prerequisite and installation. <br>
Make sure that you have the latest version of Vitis-AI and the [WeGO Example Recipes](https://github.com/Xilinx/Vitis-AI/tree/master/examples/wego#prepare-wego-example-recipes) are downloaded. <br>
Follow [the preparation step](https://github.com/Xilinx/Vitis-AI/tree/master/examples/wego#preparation) to get this done.
2. Setup Conda Environment for WeGO-Torch. <br>
Suppose you have entered the Vitis-AI CPU docker container, then using following command to activate the conda env for WeGO-Torch. <br>
    ```bash
    $ conda activate vitis-ai-wego-torch
    ```
3. Change directory to the corresponding classification folder in the WeGO folder. <br>
    ```bash
    $ cd ./pytorch/01_compiling_offline_quantized_models/classification/
    ```
4. Install the python dependencies. <br>
    ```
    $ pip install -r requirements.txt
    ```
5. Run the WeGO tool. <br>
Since we utilize the InceptionV3 pre-saved model weights from the WeGORecipes. <br>
For the example, two different running modes can be selected to enable accuracy and performance test purpose with different running options provided. <br>
   - **normal**: example will accept one single image as input and then perform the normal inference process using single thread. The output result of this mode will be either top-5 accuracy or an image, which is decided by the model type. <br>
       ```bash
       $ bash run.sh inception_v3 normal
       ```
   - **perf**: example will accept one single image as input but a large image pool will be created instead (i.e. copying the input image many times). The performance profiling process will accept this large image pool as input and then run using multi-threads. The output result of this mode will be the performance profiling result(i.e. the FPS numbers). <br> 
       ```bash
       $ bash run.sh inception_v3 perf
       ```
>  **_NOTE:_**  You may also enable the OnBoard tool option in the `run.sh`, it will collect data during the inference process, allowing to visualize using TensorBoard. 

6. Collect the WeGO artifacts in the `./_wego_torch` directory. <br>
Generated `.xmodel` artifact and meta files are saved for further deploy in the DPU device.

# Reference

Please refer to Vitis-AI UG1414 for more details about APIs usage of WeGO.

# License
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
