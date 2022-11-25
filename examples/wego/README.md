<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


# Overview

WeGO (<u>W</u>hol<u>e</u> <u>G</u>raph <u>O</u>ptimizer) is a new experimental feature within Vitis AI that aims to offer the smooth workflow for users to deploy models on cloud DPUs.  This is accomplished by integrating the Vitis AI Development with the TensorFlow 1.x, TensorFlow 2.x and PyTorch frameworks.  In this context, the native framework is employed to execute subgraphs that cannot be deployed on the cloud DPU.

WeGO automatically performs subgraph partitioning on models quantized by Vitis AI quantizer, and applies optimizations and acceleration for the cloud DPU compatible subgraphs.  Remaining portions of the graph are dispatched to the native framework for CPU execution. WeGO takes care of whole graph optimization, compilation, as well as run-time subgraph dispatch and execution. This whole process is completely transparent to the end user, which makes it very easy to use.

WeGO provides a Python programming interface that can be used to deploy models from different frameworks. This makes it possible to maximally reuse the Python code (including pre-processing and post-processing) developed during model training.  The use of WeGO greatly accelerates models deployment and evaluation on cloud DPUs.


# Preparation

## Host Environment Setup for Cloud
Before running the examples, please follow the [setup instructions for VCK5000](../../setup/vck5000) to prepare the host environment for VCK5000 PROD.  Verify that you can run Vitis-AI CPU docker container successfully.

## Prepare WeGO Example Recipes
Inside the Vitis AI Docker container, download and prepare the WeGO examples recipes (ie. models and images) by executing:
```bash
$ wget -O wego_example_recipes.tar.gz http://www.xilinx.com/bin/public/openDownload?filename=wego_example_recipes.tar.gz
$ tar xf wego_example_recipes.tar.gz -C /tmp
```

# Run WeGO Examples

Next, refer to the following links to run the wego demos targeting different frameworks according to corresponding instructions.

- [PyTorch](./pytorch) 
- [TensorFlow 2.x](./tensorflow-2.x)
- [TensorFlow 1.x](./tensorflow-1.x)

# Reference

Please refer to the [Vitis-AI User Guide, UG1414](../../docs/#release-documentation) for additional details on WeGO APIs.

# License
Copyright 2022 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License [here](../../LICENSE)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
