# ONNXRuntime with Xilinx Vitis-AI acceleration

Microsoft ONNXRuntime is a framework designed for high performance execution of ONNX models on a variety of platforms.

ONNXRuntime is enabled with Vitis-AI and available through the Microsoft github page:

https://github.com/microsoft/onnxruntime

Vitis-AI documentation for ONNXRuntime is available [here](https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/Vitis-AI-ExecutionProvider.md).

The current Vitis-AI execution provider inside ONNXRuntime enables acceleration of Neural Network model inference using DPUCADX8G. DPUCADX8G is a hardware accelerator for Convolutional Neural Networks (CNN) on top of the Xilinx Alveo platform and targets U200 and U250 accelerator cards.

## Setup

1. Follow setup instructions here to setup the ONNXRuntime - Vitis-AI environment [here](https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/Vitis-AI-ExecutionProvider.md).
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

After you have followed the above setup steps, you can copy the python image classification example and the 'images' directory containing the 'dog.jpg' test image inside the ONNXRuntime - Vitis-AI docker to run the script (for example clone this repo inside the docker container to get the script and test image). Then execute the script with:

```
python3 image_classification_DPUCADX8G.py
```

After the model has been quantized and compiled using the first N inputs you should see accelerated execution of the 'images/dog.jpg' image with the DPUCADX8G accelerator.
