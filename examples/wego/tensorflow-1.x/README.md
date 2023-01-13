# WeGO Tensorflow1 Examples Overview

Currently there are two kinds of WeGO Tensorflow1 examples:

## Compile quantized model and run it

These examples use pre-quantized models compiled into WeGO TF1 models that can be executed on the DPU. Follow the instructions in the link below to execute each example.

* [01_compiling_offline_quantized_models](./01_compiling_offline_quantized_models) 

## Quantized an float model then compile it

This example gives two action scripts.

1. The script `quantize_compile_run.sh` gives quantization flow from the float model and then compiles it into WeGO TF1 models that can be executed on the DPU. Follow the instructions in the link below to do `Preparation` and `Run Resnet_v1_50 quantize and compile flow`.

2. The script `compile_serialize_run.sh` gives the serialization and deserialization flow of the WeGO TF1 model. Follow the instructions in the link below to do `Run Resnet_v1_50 serialization and deserialization flow`.

* [02_on_the_fly_quantization](./02_on_the_fly_quantization/resnet_v1_50)

# License

Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
