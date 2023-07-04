# Pytorch Examples Overview

Currently there are a couple of types of Pytorch examples:

## Compiling an Offline Quantized Model and Run It

These examples are from previous release of WeGO and demonstrate how to compile an offline quantized model(quantized with Vitis AI Quantizer) and run it:

- [01_compiling_offline_quantized_models](./01_compiling_offline_quantized_models)

## On the Fly Quantization

A new quantize API is added to WeGO since Vitis-AI 3.0 release version. You can perform Post Training Quantization(PTQ) to get a quantized model from a float model.

This part demonstrates how to quantize, compile, optionally serialize and run a model.

There's also an example to deserialize a previously compiled WeGO module and run it.

- [02_on_the_fly_quantization](./02_on_the_fly_quantization)

## CPP API Example

This example shows how to use the CPP API of WeGO to compile an offline quantized mmodel and run it.

- [03_cpp_api_example](./03_cpp_api_example)

## Accelerate End2end Performance with ZenDNN on AMD Platforms

ZenDNN library is optimized for AMD CPU architecture, targets deep learning application and framework developers with the goal of improving deep learning inference performance on AMD CPUs. 

This part includes example(s) that will demonstrate the end-to-end performance improvments achieved by leveraging ZendDNN kernels for CPU subgraph inference in WeGO on AMD platforms.

- [04_acceleration_with_zendnn](./04_acceleration_with_zendnn)

## WeGO for PyTorch 2.0 Preview

[PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/) is fully backwards compatible and continues to provide an interactive, extensible, easy to debug, and Pythonic programming environment for AI researchers, data scientists and engineers. One of its core components, [TorchDynamo](https://pytorch.org/docs/stable/dynamo/index.html), is a Python-level JIT compiler designed to enhance the performance of unmodified PyTorch programs. TorchDynamo facilitates experimentation with various compiler backends, making it easy to explore different options. 

The following is an example specifically tailored for PyTorch 2.0 preview purpose, which will showcase the flexibility of deploying models on DPUs using Vitis-AI quantizer and WeGO-Torch in PyTorch 2.0 as two custom TorchDynamo backends.

- [05_pytorch_2.0](./05_pytorch_2.0/inception_v3/)

# License

Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
