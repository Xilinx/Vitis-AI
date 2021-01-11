# Frequently Asked Questions

- [What kinds of networks can be accelerated using the Xilinx Vitis-AI](#what-kinds-of-networks-can-be-accelerated-using-the-Xilinx-vitis-ai)
- [What layers are supported for hardware acceleration](#what-layers-are-supported-for-hardware-acceleration)
- [What frameworks are supported](#what-frameworks-are-supported)
- [What batch sizes are supported](#what-batch-sizes-are-supported)
- [How does FPGA compare to CPU and GPU acceleration](#how-does-fpga-compare-to-cpu-and-gpu-acceleration)
- [I have a new trained model, where do I begin](#i-have-a-new-trained-model-where-do-i-begin)
- [Why does Vitis-AI need to compile a graph from my framework](#why-does-vitis-ai-need-to-compile-a-graph-from-my-framework)
- [What is quantization why needed does it impact accuracy](#what-is-quantization-why-is-it-needed-does-it-impact-accuracy)

## What kinds of networks can be accelerated using the Xilinx Vitis-AI?

 Generally: Classification, Detection, and Semantic Segmentation networks.
 More generally: Any network that contains convolutional and pooling layers.
 Some networks may require modification, and/or retraining to run optimally.
 This is because not all layers are supported in hardware.
 Specifically, the following networks have been thoroughly tested:
 - Inception v1 : aka GoogLeNet
 - ResNet (All lengths)
 - SqueezeNet
 - MobileNet
 - VGG16
 - YOLOv2 (Leaky ReLU -> ReLU , reorg -> maxpool, retrained)

## What layers are supported for hardware acceleration?

- Convolutional
- Elementwise Addition
- Concatenation
- Pooling (Max/Average)
- ReLU
- BatchNorm (Via Layer Fusion)
- Scale (Via Layer Fusion)

Note: Layers not supported for hardware acceleration typically are not a performance bottleneck, and therefore can be ran on the CPU as a post processing step. In example the final fully connected layer of Inception v1.

## What frameworks are supported?

The Xilinx Vitis-AI will need to compile the dataflow graph from a framework.
Currently supported frameworks are:
- [Caffe](https://caffe.berkeleyvision.org/)
- [Tensorflow](https://www.tensorflow.org/api_docs/)

Support for other frameworks is acheived via framework conversion utilities

## What batch sizes are supported?

Since FPGA hardware accelerators can be designed for EXACTLY the task at hand, there is no need to follow the GPU convention of batching or equivalently sharing weights across images and computing inference on many images in parallel.
The FPGA hardware acclerator from Xilinx ("XDNN") will process 1 image at a time.
Different FPGA configurations can have a different number of accelerators.
Typically, we will compute inference on 1,2,4, or 8 images simultaneously.
This vastly improves the application level latency incurred by waiting to accumulate a large batch of images.

## How does FPGA compare to CPU and GPU acceleration?

FPGA accelerated networks can run up to 90x faster as compared to CPU.
FPGA accelerated networks are on par with GPU accelerated networks for throughput critical applications, yet provide support for more custom applications.
FPGA accelerated networks are far superior to GPU accelerated networks for latency critical applications such as autonomous driving.

[See white paper for benchmark](https://www.xilinx.com/support/documentation/white_papers/wp504-accel-dnns.pdf)

## I have a new trained model, where do I begin?
1. Ensure the model is defined in a framework that is digestible by Xilinx Vitis-AI (Caffe, Tensorflow, Keras, MXNet, Darknet)
2. Check for existing examples of how to run that network
  a. Vitis-AI/examples/DPUCADX8G/deployment_modes
  b. Vitis-AI/examples/DPUCADX8G/notebooks
3. Ensure your software environment is set up appropriately
  a. `conda activate vitis-ai-caffe` or `conda activate vitis-ai-tensorflow`
4. Ensure your hardware environment is set up appropriately (Before you attempt to deploy on hardware)
  a. If you are running in the cloud, you can take it for granted that this is done.
  b. [Hardware setup](https://www.xilinx.com/cgi-bin/docs/bkdoc?k=vcu1525;d=ug1268-vcu1525-reconfig-accel-platform.pdf;a=xBoardInstallation)
5. [Quantize the model](./quantizer.md)
6. [Compile the model](./compiler.md)
7. Deploy the model
  a. This is accomplished using Vitis-AI DPUCADX8G python APIs.
  b. [See test classify](../../examples/DPUCADX8G/deployment_modes/test_classify.py)
  c. Alternatively, see the jupyter tutorial that covers quantize, compile, deploy for image classification w/ InceptionV1
      aa. [image classification caffe](../../examples/DPUCADX8G/notebooks/image_classification_caffe.ipynb)

Should you hit errors at step 5, it is possible that your network needs some modification to run optimally.
Ensure that you aren't dealing with unsupported layers.
Leverage the community for debug help, by accessing the [Vitis-AI forum](https://www.xilinx.com)

## Why does Vitis-AI need to compile a graph from my framework?

The Xilinx "XDNN" hardware accelerator implements a command queue, which holds instructions used to execute an inference.
This allows the accelerator to run an entire inference instead of running layer by layer.
This implementation minimizes data movement between the host CPU, and the FPGA.

## What is quantization? Why is it needed? Does it impact accuracy?

Quantization is a very old concept. The idea is to map a range of real numbers to a set of discrete numbers.
This is how digital audio has always worked. The magnitude of a pressure wave (aka sound) could be near infinite. Imagine the sound of two planets colliding.
However, that range does not need to be represented when you are listening to The Beatles greatest hits.
Digital Audio typically quantizes an electrical signal from a microphone into a 16b or 24b discrete numbers.

In computers, real numbers are best approximated using the floating point representation, which is in and of its self a logarithmic quantization of real values.

However, floating point arithmetic is classically known to be more complex, and more demanding of hardware resources.

To implement neural networks more efficiently in FPGAs we re-quantize images, weights, and activations into 16b or 8b integer representations.

This process requires determining the range of floating point values that need to be represented and determining a set of scaling factors.

Fixed point arithmetic allows us to reach maximum OPs/second, and there is a pool of research papers that show how fixed point quantization minimally impacts the accuracy of convolutional neural nets. Some papers citing a degradation in accuracy of 2%.

Xilinx has actually seen some networks perform better with fixed-point quantization.
