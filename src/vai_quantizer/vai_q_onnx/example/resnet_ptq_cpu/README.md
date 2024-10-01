# Vitis AI ONNX Quantization Example
This folder contains example code for quantizing a Resnet model using vai_q_onnx. The example has the following parts:

1. Prepare data and model
2. Quantization

## Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```


## Prepare data and model
To Prepare the model and test data:
```
python prepare.py
```
This will generate float model resnet_trained_for_cifar10.onnx into models/ and cifar10 dataset to data/

## Quantization

Quantization tool takes the pre-processed float32 model and produce a quantized model.

```
python resnet_ptq_example_QDQ_U8S8.py
```
This will generate quantized model using QDQ quant format and UInt8 activation type and Int8 weight type to models/resnet.qdq.U8S8.onnx

```
python resnet_ptq_example_QOperator_U8S8.py
```
This will generate quantized model using QOperator quant format and UInt8 activation type and Int8 weight type to models/resnet.qdq.U8S8.onnx


## License

Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT

