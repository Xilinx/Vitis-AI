# Frequently Asked Questions

- [What is Vitis AI](#what-is-vitis-ai)
- [What are the different components of the Vitis AI](#what-are-the-different-components-of-the-vitis-ai)
- [Are all the components of Vitis AI free](#are-all-the-components-of-vitis-ai-free)
- [Is Vitis AI a separate download](#is-vitis-ai-a-separate-download)
- [Is there a specific version compatibility between Vitis AI and Vitis](#is-there-a-specific-version-compatibility-between-vitis-ai-and-vitis)
- [Which deep learning frameworks will Vitis AI support](#which-deep-learning-frameworks-will-vitis-ai-support)
- [Which AI Models will Vitis AI Support](#which-ai-models-will-vitis-ai-support)
- [Which Vitis Platforms will Vitis AI Support](#which-vitis-platforms-will-vitis-ai-support)
- [Which AI Overlay will Vitis AI Support](#which-ai-overlay-will-vitis-ai-support)
- [What does Vitis AI Library provide](#what-does-vitis-ai-library-provide)
- [How does developer migrate to Vitis AI](#how-does-developer-migrate-to-vitis-ai)
- [How to support AI development for existing Vivado developers](#how-to-support-ai-development-for-existing-vivado-developers)
- [What layers are supported for hardware acceleration](#what-layers-are-supported-for-hardware-acceleration)
- [What batch sizes are supported](#what-batch-sizes-are-supported)
- [How does FPGA compare to CPU and GPU acceleration](#how-does-fpga-compare-to-cpu-and-gpu-acceleration)
- [Where do I begin with a new trained model](#where-do-i-begin-with-a-new-trained-model)
- [What is quantization why needed does it impact accuracy](#what-is-quantization-why-needed-does-it-impact-accuracy)


## What is Vitis AI

Vitis AI is our unified AI inference solution for all Xilinx platforms (including Versal ACAPs) from edge to cloud. 
It consists of optimized IP, tools, libraries, models, and example designs. It’s a unification of the following:  
* DNNDK (AI Inference solution for the Edge) 
* ML Suite (AI Inference solution for on-premise and cloud)
* Metropolis (AI Inference solution for Versal ACAPs) –  note “Metropolis” was only an internal project name

With Vitis AI, ML and AI developers can have a familiar and consistent user experience that’s scalable across all Xilinx deployment options. 
Visit www.xilinx.com/vitis-ai for an overview

 


## What are the different components of the Vitis AI

Please refer to the overview and description of the different components on the Vitis AI Product Page here: www.xilinx.com/vitis-ai
 
## Are all the components of Vitis AI free
Everything except Vitis AI Optimizer is completely free. Vitis AI Optimizer does require a separate license. Please contact your Xilinx representative for additional information on this.

## Is Vitis AI a separate download
Yes, it is.
The Vitis AI development environment will not be part of the Vitis core development kit download through the Vitis Installer. 

## Is there a specific version compatibility between Vitis AI and Vitis
Vitis AI will have a new release every 3-months. 
Each Vitis AI release will be compatible with the most current version of Vitis tools – E.g. Vitis AI 1.0 is compatible with Vitis tools 2019.2.


## Which deep learning frameworks will Vitis AI support  

At launch (Vitis AI 1.0), we will have support for Tensorflow, Caffe and Darknet. We will continue to add support for other popular frameworks such as Pytorch in near future. 

## Which AI Models will Vitis AI Support
The models supported by Vitis AI will be published in [Vitis AI Model Zoo](https://github.com/Xilinx/AI-Model-Zoo),  
where you can find model files, model information and model performance number on various platforms.
Vitis AI also supports other models including custom models, which may not be in the Model Zoo yet. 


## Which Vitis Platforms will Vitis AI Support
Vitis AI will support Zynq-7000 and ZU+ embedded platforms such as ZCU102, ZCU104, Ultra96, Zedboard and Alveo acceleration cards such as U50, U200, U250, U280.  
At launch (Vitis AI 1.0), we will support ZCU102, ZCU104, U200, U250. It will also support ACAP platforms in near future.  
It is also designed to support AI application development on custom Vitis platforms from customers and partners.     

## Which AI Overlay will Vitis AI Support
At launch (Vitis AI 1.0), DPU for Zynq-7000 and ZU+ MPSoC and DPU for Alveo (former xDNNv3) will be supported.   
The upgraded DPU for Alveo will be supported as early access.   
Different versions of DPU will be supported to favor throughput or latency with HBM or non-HBM platforms.   
CNN Overlay for Versal ACAP and non-CNN overlays such as LTSM, MLP, NLP is under development or planned to be supported.  
For detailed roadmap, please contact your Xilinx representative.

## What does Vitis AI Library provide
Vitis AI Library is the evolution of previous AI SDK.  
It provides a lightweight set of C++ APIs across different AI tasks, such as image classification, object detection, semantic segmentation and etc.  
It enables easier application development with optimized software codes.   
It also provides memory management, and interrupt handling based on XRT.   

## How does developer migrate to Vitis AI
It will be the similar developing flow using AI quantizer and AI compiler as before.  
For the legacy AI development based on DNNDK/xfDNN, the deployment source code can be migrated to Vitis AI easily.  
What the developer must take care is to change Vivado DPU to Vitis DPU, which will be released together with Zynq DPU TRD on Xilinx GitHub.  
For the request on upgraded DPU for Alveo, please contact your Xilinx representative.

## How to support AI development for existing Vivado developers
For Vivado-based AI development, it will still be available as to follow the existing flow of Vivado DPU integration.  
DPU for Vivado will be maintained and updated along with Vitis AI 1.0 and thereafter.     


## What layers are supported for hardware acceleration

- Convolution
- Depthwise convolution
- Deconvolution
- Elementwise Addition
- Concatenation
- Pooling (Max/Average)
- ReLU / Leaky Relu / Relu6
- Fully Connected
- BatchNorm 
- Scale 
- Reorg
- Softmax

Note: Layers not supported for hardware acceleration typically are not a performance bottleneck, and therefore can be ran on the CPU as a post processing step. 


## What batch sizes are supported

Since FPGA hardware accelerators can be designed for EXACTLY the task at hand, there is no need to follow the GPU convention of batching or equivalently sharing weights across images and computing inference on many images in parallel.  
The FPGA hardware acclerator from Xilinx will process 1 image per core at a time.  
Different FPGA configurations can have a different number of accelerators.  
Typically, we will compute inference on 1,2,4, or 8 images simultaneously.  
This vastly improves the application level latency incurred by waiting to accumulate a large batch of images.  

## How does FPGA compare to CPU and GPU acceleration

FPGA accelerated networks can run upto 90x faster as compared to CPU.  
FPGA accelerated networks are on par with GPU accelerated networks for throughput critical applications, yet provide support for more custom applications.  
FPGA accelerated networks are far superior to GPU accelerated networks for latency critical applications such as autonomous driving.

[See white paper for benchmark](https://www.xilinx.com/support/documentation/white_papers/wp504-accel-dnns.pdf)

## Where do I begin with a new trained model

1. Ensure the model is defined in a framework that is digestible by Vitis AI (Caffe, Tensorflow, Darknet)
2. Ensure your software environment is set up appropriately  
  a. [docker setup](./install_docker/load_run_docker.md) 
3. Ensure your hardware environment is set up appropriately (Before you attempt to deploy on hardware)  
4. Quantize the model  
5. Compile the model  
6. Deploy the model  
  a. This is accomplished using Vitis AI C++ or Python APIs.  
  b. For Alveo, [See Vitis AI samples](../alveo/examples/vitis_ai_alveo_samples)  
  c. For ZU+ MPSoC, [See Vitis AI samples](../mpsoc//mpsoc/vitis_ai_samples_zcu102)

Should you hit errors at step 5, it is possible that your network needs some modification to run optimally.  
Ensure that you aren't dealing with unsupported layers.  
Leverage the community for debug help, by accessing the [Vitis AI Forum](https://forums.xilinx.com/t5/Machine-Learning/bd-p/Deephi)


## What is quantization why needed does it impact accuracy

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

