# Pytorch Examples Overview

Currently there are two kinds of Pytorch examples:

## Compiling an Offline Quantized Model and Run It

These examples are from previous release of WeGO and demonstrate how to compile an offline quantized model(quantized with Vitis AI Quantizer) and run it:

- [01_compiling_offline_quantized_models](./01_compiling_offline_quantized_models)

## On the Fly Quantization

A new quantize API is added to WeGO since Vitis-AI 3.0 release version. You can perform Post Training Quantization(PTQ) to get a quantized model from a float model.

This example demonstrate how to quantize, compile, optionally serialize and run a model.

There's also an example to deserialize a previously compiled WeGO module and run it.

- [02_on_the_fly_quantization](./02_on_the_fly_quantization)

# License

Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.