# Overview
 
This is a customized tf2onnx tool used for Vitis-AI. It is forked from [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx/tree/r1.13).
 
Compared to the official repo, this specially customized to support converting Vitis-AI vai_q_tensorflow1.x and vai_q_tensorflow2.x quantized models to onnx format.

# Installation
 
```bash
$ sh build.sh
$ pip install --upgrade dist/tf2onnx-1.13.0_vitis_ai-py3-none-any.whl
```

# License
[Apache License v2.0](LICENSE)
