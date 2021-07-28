# ONNX Runtime with Xilinx Vitis AI DPU acceleration

Microsoft ONNX Runtime is a framework designed for high performance execution of ONNX models on a variety of platforms.

ONNX Runtime is enabled with Vitis AI and available through the Microsoft [ONNX Runtime](https://github.com/microsoft/onnxruntime) github page.

Vitis AI documentation for ONNX Runtime is available [here](https://www.onnxruntime.ai/docs/reference/execution-providers/Vitis-AI-ExecutionProvider.html).

The current Vitis AI execution provider inside ONNX Runtime enables acceleration of Neural Network model inference with DPUCAHX8H (U50/U280) and DPUCZDX8G (Zynq) DPUs. These DPUs are hardware accelerators for Convolutional Neural Networks (CNN) on top of the Xilinx Alveo and Zynq platforms.

## Setup

1. Follow [ONNX Runtime - Vitis AI ExecutionProvider] setup instructions.
2. Download minimal ImageNet validation dataset (step specific to this example):
   ```
   python3 -m ck pull repo:ck-env
   python3 -m ck install package:imagenet-2012-val-min
   ```
3. Install ONNX and Pillow packages
   ```
   pip3 install --user onnx pillow
   ```
4. (Optional) set the number of inputs to be used for on-the-fly quantization to a lower number (e.g. 8) to decrease the quantization time (potentially at the cost of lower accuracy):
   ```
   export PX_QUANT_SIZE=8
   ```

## Run example

After you have followed the above setup steps, you can copy the python image classification example and the 'images' directory containing the 'dog.jpg' test image inside the ONNX Runtime - Vitis AI docker to run the script (for example clone this repo inside the docker container to get the script and test image). Then execute the script with:

```
python3 image_classification_DPUCAHX8H_U50.py
```

After the model has been quantized and compiled using the first N inputs you should see accelerated execution of the 'images/dog.jpg' image with the DPUCAHX8H accelerator.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job.)

   [ONNX Runtime - Vitis AI ExecutionProvider]: https://www.onnxruntime.ai/docs/reference/execution-providers/Vitis-AI-ExecutionProvider.html
