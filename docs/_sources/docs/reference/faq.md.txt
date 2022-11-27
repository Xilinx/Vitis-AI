# Frequently Asked Questions

- [What is Vitis AI?](#what-is-vitis-ai)
- [What are the different components of Vitis AI?](#what-are-the-different-components-of-vitis-ai)
- [Are all the components of Vitis AI free?](#are-all-the-components-of-vitis-ai-free)
- [Is Vitis AI a separate download?](#is-vitis-ai-a-separate-download)
- [What are the Vitis AI, Vitis, and Vivado version compatibility requirements?](#what-are-the-vitis-ai-vitis-and-vivado-version-compatibility-requirements)
- [Which deep learning frameworks does Vitis AI support?](#which-deep-learning-frameworks-does-vitis-ai-support)
- [Which AI Models does Vitis AI Support?](#which-ai-models-will-vitis-ai-support)
- [Which Vitis Platforms will Vitis AI Support?](#which-vitis-platforms-does-vitis-ai-support)
- [What does the Vitis AI Library provide?](#what-does-the-vitis-ai-library-provide)
- [What layers are supported for hardware acceleration?](#what-layers-are-supported-for-hardware-acceleration)
- [What batch sizes are supported?](#what-batch-sizes-are-supported)
- [How does FPGA compare to CPU and GPU acceleration?](#how-does-fpga-compare-to-cpu-and-gpu-acceleration)
- [Where do I begin with a new trained model?](#where-do-i-begin-with-a-new-trained-model)
- [What is quantization why needed does it impact accuracy?](#what-is-quantization-why-needed-does-it-impact-accuracy)


## What is Vitis AI?

Vitis AI is our unified AI inference solution for all Xilinx platforms.  It consists of optimized IP, tools, libraries, models, and example designs.

With Vitis AI, ML and AI developers can have a familiar and consistent user experience that is scalable from edge-to-cloud across a variety of Xilinx targets.

## What are the different components of Vitis AI?

To address this question, we would suggest you review the workflow documentation in this repo.  In addition, you may refer to the overview and description of the different components on the Vitis AI Product Page here: www.xilinx.com/vitis-ai

## Are all the components of Vitis AI free?
Everything except Vitis AI Optimizer is free. The Vitis AI Optimizer does require a separate license. Additional details surrounding this license can be found in the [Optimizer section of the Introductory documentation](../README.md#model-optimization).

## Is Vitis AI a separate download?
Since you have arrived at this Github repository, you have probably already deduced the answer to this question 😂.  However, the answer is yes, you can get started by cloning this repository.

## What are the Vitis AI, Vitis, and Vivado version compatibility requirements?
Vitis AI, Vitis and Vivado are released on a bi-annual cadence.  Currently there is a slight lag in the release timing of Vitis AI in order to address the complexities involved in verification and compatibility.  Each Vitis AI release is verified with the (then) current release of Vivado and Vitis.  See [here](../README.md#version-compatibility) for more details.

## Which deep learning frameworks does Vitis AI support?

Vitis AI 2.5 supports TensorFlow 1.x, 2.x and PyTorch.  Prior to release 2.5, Caffe and DarkNet were supported.  For Caffe and DarkNet support, users can leverage a previous release.

## Which AI Models does Vitis AI Support?
The answer is that in general, Vitis AI can support the majority of CNNs, including custom user networks.  As part of our development process and continuous effort to provide more diverse operator and layer support, we train and deploy new models with each release.  All of these models, together with performance benchmarks for out-of-the-box supported Xilinx targets are published in the [Vitis AI Model Zoo](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/model_zoo).

## What Xilinx Target Device Families and Platforms does Vitis AI Support?
Vitis AI DPUs are available for both Zynq Ultrascale+ MPSoC as well as Versal Edge and Core chip-down designs.  The Kria K26 SOM is supported as a production-ready Edge platform, and Alveo accelerator cards are supported for cloud applications.

## What does the Vitis AI Library provide?
The Vitis AI Library provides a lightweight set of C++ APIs suited to a variety of AI tasks, such as image classification, object detection and semantic segmentation, simplifying deployment of user applications.

## What layers are supported for hardware acceleration?
The answer to this question varies somewhat depending on the specific target platform and DPU IP that is leveraged on that platform.  

In general, most common CNN layers and activation types are supported for DPU hardware acceleration.  The toolchain supports graph partitioning, which enables developers to augment DPU operator support through the use of an inference framework, or via custom acceleration code.

More specific details can be found [here](../workflow-model-development#operator-support)

## What batch sizes are supported?

Since FPGA hardware accelerators can be designed for precisely for the task at hand, the notion of batching as it is widely understood in CPU and GPU deployments is not directly applicable to Xilinx targets.  In general, Xilinx DPUs process one image per accelerator core at a time.  The implication of this is that the performance of a single-core DPU (Zynq/Kria targets) is specified with a batch size of 1, and that inference efficiency is not improved through the use of batching.

For Versal and Alveo targets, higher performance DPUs have been developed that have more than one core, and in this context, the runtime and DPU are designed to compute inference on multiple images simultaneously.  Thus, in the case of Versal and Alveo targets, batching is leveraged in order to ensure that each of the parallel acceleration cores is performing useful processing.  However, this also differs from the CPU/GPU notion of batching in that the batch size requirement is much lower (2-8).  For specific details, please refer to the respective [DPU product guides](../README.md/#release-documentation)

As there is a greatly reduced requirement to queue multiple input samples, the end-to-end latency and memory footprint for inference is reduced, which can be an important advantage in some applications.

## How does FPGA compare to CPU and GPU acceleration?

FPGA accelerated networks can run upto 90x faster as compared to CPU.  
FPGA accelerated networks are on par with GPU accelerated networks for throughput critical applications, yet provide support for more custom applications.  
FPGA accelerated networks are far superior to GPU accelerated networks for latency critical applications such as autonomous driving.

[See white paper for benchmark](https://www.xilinx.com/support/documentation/white_papers/wp504-accel-dnns.pdf)

## Where do I begin with a new trained model?

We would recommend the workflow documentations pages in this repository as the ideal starting point for new users.  

## What is quantization why needed does it impact accuracy?

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
