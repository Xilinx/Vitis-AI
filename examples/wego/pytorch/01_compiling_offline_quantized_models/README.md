# Setup Conda Environment for WeGO-Torch

Suppose you have entered the Vitis-AI CPU docker container, then using following command to activate the conda env for WeGO-Torch.

```bash
$ conda activate vitis-ai-wego-torch
```

# Runing the Examples

## Running Mode

For each example, two different running modes can be selected to enable accuracy and performance test purpose with different running options provided.

- **normal** : example will accept one single image as input and then perform the normal inference process using single thread. The output result of this mode will be either top-5 accuracy or an image, which is decided by the model type. 

- **perf** : example will accept one single image as input but a large image pool will be created instead(i.e. copying the input image many times). The performance profiling process will accept this large image pool as input and then run using multi-threads. The output result of this mode will be the performance profiling result(i.e. the FPS numbers).

Please refer to each example link for more details.

## How to Run

After the wego-torch conda environment is activated, please refer to following links to run the examples according to corresponding instructions. 

- [Inception_v3](./classification) 
- [SqueezeNet](./classification)
- [ofa_resnet50_0_9B](./ofa_resnet50_0_9B)
- [drunet](./drunet)

# License

Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
