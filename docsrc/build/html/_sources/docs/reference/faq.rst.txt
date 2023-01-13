==========================
Frequently Asked Questions
==========================

-  `What is Vitis AI? <#what-is-vitis-ai>`__
-  `What are the different components of Vitis
   AI? <#what-are-the-different-components-of-vitis-ai>`__
-  `Are all the components of Vitis AI
   free? <#are-all-the-components-of-vitis-ai-free>`__
-  `Is Vitis AI a separate
   download? <#is-vitis-ai-a-separate-download>`__
-  `What are the Vitis AI, Vitis, and Vivado version compatibility
   requirements? <#what-are-the-vitis-ai-vitis-and-vivado-version-compatibility-requirements>`__
-  `Which deep learning frameworks does Vitis AI
   support? <#which-deep-learning-frameworks-does-vitis-ai-support>`__
-  `Which AI Models does Vitis AI
   Support? <#which-ai-models-will-vitis-ai-support>`__
-  `Which Vitis Platforms will Vitis AI
   Support? <#which-vitis-platforms-does-vitis-ai-support>`__
-  `What does the Vitis AI Library
   provide? <#what-does-the-vitis-ai-library-provide>`__
-  `What layers are supported for hardware
   acceleration? <#what-layers-are-supported-for-hardware-acceleration>`__
-  `What batch sizes are supported? <#what-batch-sizes-are-supported>`__
-  `How does FPGA compare to CPU and GPU
   acceleration? <#how-does-fpga-compare-to-cpu-and-gpu-acceleration>`__
-  `Where do I begin with a new trained
   model? <#where-do-i-begin-with-a-new-trained-model>`__
-  `What is quantization why needed does it impact
   accuracy? <#what-is-quantization-why-needed-does-it-impact-accuracy>`__
-  `Can I leverage Vitis™ AI on a pure FPGA target with or without a Microblaze™ processor? <#can-i-leverage-vitis-ai-on-a-pure-fpga-target-with-or-without-a-microblaze-processor>`__
-  `Is it possible to use the DPU without PetaLinux? <#is-it-possible-to-use-the-dpu-without-petalinux>`__
-  `Is it possible to deploy the DPUCZ without using Linux? <#is-it-possible-to-use-the-dpu-without-petalinux>`__
-  `Can the DPUCZ be used for alternate purposes beyond deployment of neural networks? For example, signal processing operations? <#can-the-dpucz-be-used-for-alternate-purposes-beyond-deployment-of-neural-networks-for-example-signal-processing-operations>`__
-  `What is the difference between the Vitis AI integrated development environment and the FINN workflow? <#what-is-the-difference-between-the-vitis-ai-integrated-development-environment-and-the-finn-workflow>`__
-  `I have a ZCU106 board. Can I leverage the Vitis AI IDE with the ZCU106? How do I get started? <#i-have-a-zcu106-board-can-i-leverage-the-vitis-ai-ide-with-the-zcu106-how-do-i-get-started>`__
-  `What is the specific AI accelerator that AMD Xilinx provides for Zynq™ Ultrascale+? Is it a systolic array? <#what-is-the-specific-ai-accelerator-that-amd-xilinx-provides-for-zynq-ultrascale-is-it-a-systolic-array>`__




What is Vitis AI?
-----------------

Vitis AI is our unified AI inference solution for all Xilinx platforms. It consists of optimized IP, tools, libraries, models, and example
designs. With Vitis AI, ML and AI developers can have a familiar and consistent user experience that is scalable from edge-to-cloud across a variety of Xilinx targets.

Are all the components of Vitis AI free?
----------------------------------------

Everything except Vitis AI Optimizer is free. The Vitis AI Optimizer does require a separate license. Additional details surrounding this
license can be found in the Optimizer :ref:`introduction <model_optimization>`.

Is Vitis AI a separate download?
--------------------------------

Yes!  Users can started by cloning the Vitis AI Github `repository <https://github.com/Xilinx/Vitis-AI>`__

What are the Vitis AI, Vitis, and Vivado version compatibility requirements?
----------------------------------------------------------------------------

Vitis AI, Vitis and Vivado are released on a bi-annual cadence. Currently there is a slight lag in the release timing of Vitis AI in order to address the complexities involved in verification and compatibility with Vivado® and Vitis™. Each Vitis AI release is verified with the (then) current release of Vivado and Vitis. It is not generally advised to mix tool versions.  See :doc:`the version compatibility documentation <version_compatibility>` for more details.

Which deep learning frameworks does Vitis AI support?
-----------------------------------------------------

Vitis AI 3.0 supports TensorFlow 1.x, 2.x and PyTorch. Prior to release 2.5, Caffe and DarkNet were supported and for those frameworks, users can leverage a previous release of Vitis AI for quantization and compilation, while leveraging the latest Vitis-AI Library and Runtime components for deployment.

Which AI Models does Vitis AI Support?
--------------------------------------

With the release of the Vitis AI IDE, more than 120 models are released in the Vitis AI Model Zoo. In addition, the Vitis AI IDE is designed to enable developers to deploy custom models, subject to layer, parameter, and activation support. Due to the GPL licensing associated with newer YOLO variants, we will not be releasing pre-trained versions of these models. Users will need to train, quantize,and compile those models using the Vitis AI tool, and we will provide the resources to enable this. In general, Vitis AI can support the majority of CNNs, including custom user networks. As part of our development process and continuous effort to provide more diverse operator and layer support, we train and deploy new models with each release. All of these models, together with performance benchmarks for out-of-the-box supported Xilinx targets are published in the :ref:`Vitis AI Model Zoo <workflow-model-zoo>`.

What Xilinx Target Device Families and Platforms does Vitis AI Support?
-----------------------------------------------------------------------

Vitis AI DPUs are available for both Zynq Ultrascale+ MPSoC as well as Versal Edge and Core chip-down designs. The Kria K26 SOM is supported as a production-ready Edge platform, and Alveo accelerator cards are supported for cloud applications. 

What does the Vitis AI Library provide?
---------------------------------------

The Vitis AI Library provides a lightweight set of C++ APIs suited to a variety of AI tasks, such as image classification, object detection and semantic segmentation, simplifying deployment of user applications.

What layers are supported for hardware acceleration?
----------------------------------------------------

The answer to this question varies somewhat depending on the specific target platform and DPU IP that is leveraged on that platform. In general, most common CNN layers and activation types are supported for DPU hardware acceleration. The toolchain supports graph partitioning, which enables developers to augment DPU operator support through the use of an inference framework, or via custom acceleration code.

More specific details can be found
:ref:`here <operator-support>`

What batch sizes are supported?
-------------------------------

Since FPGA hardware accelerators can be designed for precisely for the task at hand, the notion of batching as it is widely understood in CPU and GPU deployments is not directly applicable to Xilinx targets. In general, Xilinx DPUs process one image per accelerator core at a time. The implication of this is that the performance of a single-core DPU (Zynq/Kria targets) is specified with a batch size of 1, and that inference efficiency is not improved through the use of batching. 

For Versal and Alveo targets, higher performance DPUs have been developed that have more than one core, and in this context, the runtime and DPU are designed to compute inference on multiple images simultaneously. Thus, in the case of Versal and Alveo targets, batching is leveraged in order to ensure that each of the parallel acceleration cores is performing useful processing. However, this also differs from the CPU/GPU notion of batching in that the batch size requirement is much lower (2-8). For specific details, please refer to the respective :doc: `DPU product guides <release_documentation>`

As there is a greatly reduced requirement to queue multiple input samples, the end-to-end latency and memory footprint for inference is reduced, which can be an important advantage in some applications.

How does FPGA compare to CPU and GPU acceleration?
--------------------------------------------------

FPGA accelerated networks can run upto 90x faster as compared to CPU. FPGA accelerated networks are on par with GPU accelerated networks for throughput critical applications, yet provide support for more custom applications. FPGA accelerated networks are far superior to GPU accelerated networks for latency critical applications such as autonomous driving. 
`See this white paper for an example benchmark <https://www.xilinx.com/support/documentation/white_papers/wp504-accel-dnns.pdf>`__

Where do I begin with a new trained model?
------------------------------------------

We would recommend the workflow documentations pages in this repository as the ideal starting point for new users.

What is quantization why needed does it impact accuracy?
--------------------------------------------------------

Quantization is a very old concept. The idea is to map a range of real   numbers to a set of discrete numbers. This is how digital audio has always worked. The magnitude of a pressure wave (aka sound) could be near infinite. Imagine the sound of two planets colliding. However, that range does not need to be represented when you are listening to The Beatles greatest hits. Digital Audio typically quantizes an electrical signal from a microphone into a 16b or 24b discrete numbers. In computers, real numbers are best approximated using the floating point representation, which is in and of its self a logarithmic quantization of real values. However, floating point arithmetic is classically known to be more complex, and more demanding of hardware resources. To implement neural networks more efficiently in FPGAs we re-quantize images, weights, and activations into 16b or 8b integer representations. This process requires determining the range of floating point values that need to be represented and determining a set of scaling factors. Fixed point arithmetic allows us to reach maximum OPs/second, and there is a pool of research papers that show how fixed point quantization minimally impacts the accuracy of convolutional neural nets. Some papers citing a degradation in accuracy of 2%. Xilinx has actually seen some networks perform better with fixed-point quantization. 

Can I leverage Vitis™ AI on a pure FPGA target with or without a Microblaze™ processor? 
---------------------------------------------------------------------------------------

The Vitis AI integrated development environment (IDE) supports SoC targets (Zynq™UltraScale+™ MPSoC, Versal™ ACAP) and Alveo™ platforms (AMD64 host). It does not claim to support FPGA-class devices including Spartan™, Artix™, Kintex™, or Virtex™ FPGAs.  While it is possible to enable and run Vitis AI IDE firmware components on the MicroBlaze processor, this is not a documented and supported flow for mainstream development. Officially, the AMD Xilinx “Space deep-learning processor unit (DPU)” project leverages the MicroBlaze processor targeting Kintex UltraScale-class devices. For deployment in standard commercial applications, we do have an experimental flow that we can potentially share, but it has limitations, and the expectation is that the developer will need to invest additional time in optimization. If you have a strong need for this, please reach out to us directly, and we can discuss your use case further.

Is it possible to use the DPU without PetaLinux? 
------------------------------------------------

There are at least two potential interpretations of this question:

`Is it possible to deploy the DPUCZ using Yocto flows, or even Ubuntu, rather than PetaLinux?`

	Yes, what is important to consider is that each release of the Vitis AI tool and the DPUCZ IP is provided with drivers and a runtime that targets a specific Linux kernel release. Misalignment between the target kernel version can pose challenges and may require extensive code changes.

`Is it possible to deploy the DPUCZ without using Linux?` 

	We do have proof-of-concept beta support for both Green Hills Integrity and Blackberry QNX. For QNX and Integrity support, users should contact their local FAE or Sales representative to request additional discussions with the factory. As of this release, no support exists for bare-metal or FreeRTOS; however, Zynq UltraScale+ family members do support asymmetric multiprocessing, with the potential that developers can integrate the DPU via Linux, while continuing to develop the bulkof their application in their chosen OS. Please refer to `UG1137 <https://docs.xilinx.com/r/en-US/ug1137-zynq-ultrascale-mpsoc-swdev/Asymmetric-Multiprocessing-AMP>`__ for additional information on AMP modes of operation.

Can the DPUCZ be used for alternate purposes beyond deployment of neural networks? For example, signal processing operations?
-----------------------------------------------------------------------------------------------------------------------------

While Xilinx DPUs are well optimized for certain operations that overlap with signal processing (i.e., convolution, elementwise, etc.), deployment of conventional signal processing functions is neither the purpose nor intent of the Vitis AI IDE and the DPUCZ. Today, the Vitis AI tool DPU instruction compiler is not provided as open source, and the instruction set for the DPUCZ is not publicly documented.For signal processing applications, we provide highly optimized IP cores as well as open-source libraries supporting a wide variety of FFT architectures, including both streaming and time-division multiplex applications. Furthermore, we have support for advanced FFT architectures such as SSR. It is much more efficient to deploy such functions by leveraging IP that has been optimized for these tasks.  Remember, the DPUCZ is optimized for INT8 quantized operations. In many signal processing applications, the pipeline employs a higher dynamic range, such as 10, 12, 14, bit  s.   Furthermore, many signal processing applications employ a streaming data pipeline that does not align well with the operation of the DPUCZ.Similarly, common signal processing operations are provided as optimized IP in the Vivado®IP catalog. For the highest levels of performance in RTL designs, we would generally refer users to these IP. Most of the IP that you will require is free and available in the Vivado Design Suite (free “included” license).  For users who prefer an open-source, HLS-based implementation or pure software (Vitis™ IDE) flow, the Vitis Accelerated Libraries are an excellent solution.Finally, if you are an avid Simulink® tool user, you may also wish to consider our Vitis Model Composer / System Generator workflow.

What is the difference between the Vitis AI integrated development environment and the FINN workflow?
-----------------------------------------------------------------------------------------------------

The two methods are complementary. The FINN workflow is differentiated from the Vitis AI IDE workflow in that it does not employ a general-purpose AI inference accelerator. Rather, the FINN toolchain builds a network-specific dataflow architecture, leveraging streaming interfaces between layers. Effectively, the entire CNN is unrolled and implemented layer by layer in fabric. The result is that a FINN implementation is optimized specifically for a specific neural network architecture. A new bitstream is required if the user chooses to modify the structure or parameters (excluding weights) of the neural network graph. The benefits of the FINN approach can include:

- Highly optimized latency
- Highly optimized programmable logic utilization for small networks
- High throughput for small networks
- High power efficiency
- Flexibility to  employ mixed precision (quantization on a layer-by-layer basis) 

This is all in contrast to the Vitis AI DPUs, which are fixed architecture, general-purpose AI accelerators that can deploy a wide variety of neural networks without requiring a new bitstream when the neural network graph structure is changed. The benefits of the Vitis AI IDE workflow can include:

- Flexibility to deploy both shallow and deep neural networks with comparable programmable logic resource utilization 
- No requirement to update the bitstream when switching to a different neural network architecture
- Ability to deploy multiple networks and process a variable number of streams with a single accelerator
- Fixed INT8 precision 
- Low-latency, general-purpose applications
- Multi-network, multi-stream, deployments with a single DPU instance

I have a ZCU106 board. Can I leverage the Vitis AI IDE with the ZCU106? How do I get started?
---------------------------------------------------------------------------------------------

The ZCU106 board can be leveraged, but you are correct in concluding that the Vitis AI IDE repository does not provide an image that supports this board for evaluation purposes. You might wish to test drive DPU Pynq, which does have (currently unverified) support for the ZCU106.

In the past, developers have ported the DPUCZ reference design (formerly known as the TRD) to the ZCU106 board. We do not formally document this process today, but if you wish to pursue this, you may want to start with these references:

- `DPU-PYNQ <https://github.com/Xilinx/DPU-PYNQ/tree/master/boards/zcu106>`__
- `Xilinx Specialist, Jim Heaton's reference <https://github.com/jimheaton/Vitis-AI-DPU_TRD-for-ZCU106>`__
- `Vitis AI Design Partner, Logictronix's reference <https://logictronix.com/machine-learning-with-fpga/vitisai-dpu-dnndk/dpu-3-0-trd-for-zcu106/>`__
- `Xilinx forums reference <https://support.xilinx.com/s/question/0D52E00006hpM10SAE/vitisai-dputrd-for-zcu106?language=en_US>`__

What is the specific AI accelerator that AMD Xilinx provides for Zynq™ Ultrascale+?  Is it a systolic array?
------------------------------------------------------------------------------------------------------------

The DPUCZ IP that is provided with the Vitis AI IDE is the specialized accelerator. It is a custom processor that has a specialized instruction set. Graph operators such as CONV, POOL, ELTWISE are compiled as instructions that are executed by the DPU. The DPUCZ bears similarities to a systolic array but has specialized micro-coded engines that are optimized for specific tasks. Some of these engines are optimized for conventional convolution, while some are optimized for tasks such as depth-wise convolution, eltwise and others. We tend to refer to the DPUCZ as a Matrix of (Heterogeneous) Processing Engines.