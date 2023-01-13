<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# Overview

WeGO (<u>W</u>hol<u>e</u> <u>G</u>raph <u>O</u>ptimizer) is a Vitis AI new experimental feature and it aims to offer the smooth solution to deploy various models on cloud DPU through integrating Vitis AI Development kit with TensorFlow 1.x, TensorFlow 2.x and PyTorch frameworks.

WeGO automatically performs subgraph partitioning for the models quantized by Vitis AI quantizer, and applies optimizations and acceleration for the cloud DPU compatible subgraphs.  And the DPU un-supported remaining parts of graph are dispatched to framework for CPU execution. WeGO takes care of the whole graph optimization, compilation and run-time subgraphs’ dispatch and execution. This whole process is completely transparent to the end users, which makes it very easy to use. 

Using WeGO is a very straightforward transition from training to inference for model designers. WeGO provides Python programming interface to deploy the quantized models over different frameworks. This makes it possible to maximumly reuse the Python code (including pre-processing and post-processing) developed during the phase of models training with frameworks, which greatly speeds up the models’ deployment and evaluation over cloud DPUs.


# Preparation

## Setup Host Environment for Cloud
Before running the examples, please follow [setup for VCK5000](https://github.com/Xilinx/Vitis-AI/tree/master/board_setup/vck5000) to set up the host env for VCK5000 PROD and make sure you have entered the Vitis-AI CPU docker container successfully and the DPU IP has been selected properly.

> Note: currently three different docker image targeting diverse AI frameworks are provided, make sure the right docker image is used for the corresponding examples running purpose(i.e. TensorFlow 1.x docker for TensorFlow 1.x WeGO examples, TensorFlow 2.x docker for TensorFlow 2.x WeGO examples, PyTorch docker for PyTorch WeGO examples).

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


# Reference

Please refer to Vitis-AI UG1414 for more details about APIs usage of WeGO.

# License
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
