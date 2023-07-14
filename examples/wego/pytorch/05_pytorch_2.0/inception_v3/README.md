# Introduction

Ensuring backward compatibility is a top priority when supporting TorchDynamo in both Quanitzer and WeGO. The goal is to provide a similar workflow and APIs, minimizing the learning curve for users and allowing them to reuse existing scripts or code for quantization and WeGO compilation with minimal modifications.

For deploying an original float model on DPU in dynamo mode, the same steps are required as in the native WeGO-Torch 1.x flow:

1. Quantize the model using Vitis-AI quantizer, while enabling dynamo mode, following the same API and workflow.
2. Compile the float model using the same WeGO-Torch compilation API with dynamo enabled.
3. Run the compiled WeGO model on the DPU.

## APIs Extension for TorchDynamo Support

### Vitis-AI NNDCT Quantizer

Quantizing a model in dynamo mode using the Vitis-AI NNDCT Quantizer is remarkably straightforward, requiring only one additional option when creating the quantizer:

`quantizer = pytorch_nndct.apis.torch_quantizer(..., dynamo = True)`

Enabling dynamo mode simply involves setting the `dynamo` parameter to `True` within the `torch_quantizer` function call. This option seamlessly incorporates the dynamo functionality into the quantization process, making it effortless to apply dynamo mode to your model.

### WeGO-Torch

Compiling a model in dynamo mode using WeGO-Torch is equally straightforward, involving just a couple of additional steps compared to the native WeGO compilation API. Here's what you need to do:

- Create a `wego_torch.DynamoCompileOptions` object and specify the `quantize_result_path` parameter, which should point to the path where the quantization results are stored:

```python
dynamo_options = wego_torch.DynamoCompileOptions(
    quantize_result_path = quant_result_path
)
```

By providing the quantization result path, you ensure that WeGO-Torch knows where to find the quantized model for compilation.

- Use the `dynamo=True` option in the `wego_torch.compile` API call, along with the `dynamo_options` parameter:

```python
wego_mod = wego_torch.compile(float_model, wego_torch.CompileOptions(
    dynamo = True,
    dynamo_options = dynamo_options,
    ...
))
```

Setting `dynamo=True` indicates that you want to perform the compilation in dynamo mode, and supplying the `dynamo_options` object allows for proper configuration.

It's important to note the following when using `wego_torch.compile` for model compilation in dynamo mode:

- Ensure that you provide the original float model (not the quantized TorchScript model) as the input model for the WeGO compile interface.
- Input shape or other metadata is not required to be provided explicitly; the compilation process will handle it internally.

# Conda Environment for WeGO-Torch2

We will utilize a seperate conda environment **vitis-ai-wego-torch2** for PyTorch 2.0 preview purpose.It is specifically intended for preview purposes, and I recommend using it solely for testing PyTorch 2.0 TorchDynamo functionality.

Assuming you have entered the Vitis-AI CPU docker container, then using following command to activate the specific conda environment for WeGO-Torch2.

```bash
$ conda activate vitis-ai-wego-torch2
```

Once you have activated the conda environment for wego-torch2, please follow the instructions below step by step to quantize, compile & run a model on the DPU in dynamo mode.

# Preparation

## Install the Python Dependencies

```
$ pip install -r requirements.txt
```

## Prepare Dataset for Quantization

If you need to quantize the model with Post Training Quantization(PTQ), a small unlabeled dataset will be needed(100-1000 images). You can use the ImageNet dataset or your own dataset.

To use ImageNet, first download it from http://www.image-net.org/. For calibration purpose, only the validation set is needed.

After downloading prepare the dataset with [the following shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).

The directory structure should look like this:

```bash
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
```

We'll only be using the validation set here(imagenet/val/).

## Get Pretained inception_v3 Model

Download it from pytorch website:

```bash
wget https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
```

# Quantize and Compile inception_v3 with Dynamo Mode

## Quantize

Use dynamo_quantize.sh to quantize the float model in dynamo mode.

Usage:

```bash
$ bash dynamo_quantize.sh path_to_pretrained_float_model path_to_imagenet_val_folder desired_quantize_result_path
```

For example:

```bash
$ bash dynamo_quantize.sh ./inception_v3_google-0cc3c7bd.pth imagenet/val/ ./dynamo_inception_v3
```

The quantized result will be saved in path './dynamo_inception_v3'

## Compile & Run

Use dynamo_compile_run.sh to compile the float model in dynamo mode with corresponding quantization result provided and run the inference on DPU.

Usage: 

```bash
$ bash dynamo_compile_run.sh path_to_pretrained_float_model path_to_the_quantized_result_path
```

For example:

```bash
$ bash dynamo_compile_run.sh ./inception_v3_google-0cc3c7bd.pth ./dynamo_inception_v3
```

# License

Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.