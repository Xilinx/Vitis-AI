..
    Copyright 2023 Advanced Micro Devices, Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


############################################################
Vitis AI
############################################################

AMD Vitis™ AI is an integrated development environment that can be leveraged to accelerate AI inference on AMD platforms. This toolchain provides optimized IP, tools, libraries, models, as well as resources, such as example designs and tutorials that aid the user throughout the development process.  It is designed with high efficiency and ease-of-use in mind, unleashing the full potential of AI acceleration on AMD Adaptable SoCs and Alveo Data Center accelerator cards.

.. figure:: docs/reference/images/VAI_IDE.PNG
   :width: 1300

   Vitis AI Integrated Development Environment Block Diagram

The Vitis |trade| AI solution consists of three primary components:

- The Deep-Learning Processor unit (DPU), a hardware engine for optimized the inferencing of ML models
- Model development tools, to compile and optimize ML models for the DPU 
- Model deployment libraries and APIs, to integrate and execute the ML models on the DPU engine from a SW application

The Vitis AI solution is packaged and delivered as follows:

- AMD open download: pre-built target images integrating the DPU
- Vitis AI docker containers: model development tools
- Vitis AI github repository: model deployment libraries, setup scripts, examples and reference designs

############################################################
Vitis AI Key Components
############################################################

****************************
Deep-Learning Processor Unit
****************************

The :ref:`Deep-learning Processor Unit (DPU) <workflow-dpu>` is a programmable engine optimized for deep neural networks. The DPU implements an efficient tensor-level instruction set designed to support and accelerate various popular convolutional neural networks, such as VGG, ResNet, GoogLeNet, YOLO, SSD, and MobileNet, among others. 

The DPU supports on AMD Zynq |trade| UltraScale+ |trade| MPSoCs, the Kria |trade| KV260, Versal |trade| and Alveo cards. It scales to meet the requirements of many diverse applications in terms of throughput, latency, scalability, and power.

AMD provides pre-built platforms integrating the DPU engine for both edge and data-center cards. These pre-built platforms allow data-scientists to start developping and testing their models without any need for HW development expertise.

For embedded applications, the DPU needs to be integrated in a custom platform along with the other programmable logic functions going in the FPGA or adaptive SoC device. HW designers can :ref:`integrate the DPU in a custom platform <integrating-the-dpu>` using either the Vitis flow or the Vivado |trade| Design Suite.


*****************
Model Development
*****************

Vitis AI Model Zoo
==================
The :ref:`Vitis AI Model Zoo <workflow-model-zoo>` includes optimized deep learning models to speed up the deployment of deep learning inference on adaptable AMD platforms. These models cover different applications, including ADAS/AD, video surveillance, robotics, and data center. You can get started with these pre-trained models to enjoy the benefits of deep learning acceleration.

Vitis AI Model Inspector
========================
The :ref:`Vitis AI Model Inspector <model-inspector>` is used to perform initial sanity checks to confirm that the operators and sequence of operators in the graph is compatible with Vitis AI. Novel neural network architectures, operators, and activation types are constantly being developed and optimized for prediction accuracy and performance. Vitis AI provides mechanisms to leverage operators that are not natively supported by your specific DPU target.

Vitis AI Optimizer
==================
The :ref:`Vitis AI Optimizer <model-optimization>` exploits the notion of sparsity to reduce the overall computational complexity for inference by 5x to 50x with minimal accuracy degradation. Many deep neural network topologies employ significant levels of redundancy. This is particularly true when the network backbone is optimized for prediction accuracy with training datasets supporting many classes. In many cases, this redundancy can be reduced by “pruning” some of the operations out of the graph. 

Vitis AI Quantizer
==================
The :ref:`Vitis AI Quantizer <model-quantization>`, integrated as a component of either TensorFlow or PyTorch, converts 32-bit floating-point weights and activations to fixed-point integers like INT8 to reduce the computing complexity without losing prediction accuracy. The fixed-point network model requires less memory bandwidth and provides faster speed and higher power efficiency than the floating-point model.

Vitis AI Compiler
=================
The :ref:`Vitis AI Compiler <model-compilation>` maps the AI quantized model to a highly-efficient instruction set and dataflow model. The compiler performs multiple optimizations; for example, batch normalization operations are fused with convolution when the convolution operator precedes the normalization operator. As the DPU supports multiple dimensions of parallelism, efficient instruction scheduling is key to exploiting the inherent parallelism and potential for data reuse in the graph. The Vitis AI Compiler addresses such optimizations.


****************
Model Deployment
****************

Vitis AI Runtime
================
The :ref:`Vitis AI Runtime <vitis-ai-runtime>` (VART) is a set of low-level API functions that support the integration of the DPU into software applications. VART is built on top of the Xilinx Runtime (XRT) amd provides a unified high-level runtime for both Data Center and Embedded targets. Key features of the Vitis AI Runtime API include:

- Asynchronous submission of jobs to the DPU.
- Asynchronous collection of jobs from the DPU.
- C++ and Python API implementations.
- Support for multi-threading and multi-process execution.


Vitis AI Library
================
The :ref:`Vitis AI Library <vitis-ai-library>`  is a set of high-level libraries and APIs built on top of the Vitis AI Runtime (VART). The higher-level APIs included in the Vitis AI Library give developers a head-start on model deployment. While it is possible for developers to directly leverage the Vitis AI Runtime APIs to deploy a model on AMD platforms, it is often more beneficial to start with a ready-made example that incorporates the various elements of a typical application, including:

- Simplified CPU-based pre and post-processing implementations.
- Vitis AI Runtime integration at an application level.


Vitis AI Profiler
=================
The :ref:`Vitis AI Profiler <vitis-ai-profiler>` profiles and visualizes AI applications to find bottlenecks and allocates computing resources among different devices. It is easy to use and requires no code changes. It can trace function calls and run time, and also collect hardware information, including CPU, DPU, and memory utilization.




.. toctree::
   :maxdepth: 3
   :caption: Setup and Install
   :hidden:
   
   Release Notes <docs/reference/release_notes_4.0>
   System Requirements <docs/reference/system_requirements>
   Host Install Instructions <docs/install/install>

.. toctree::
   :maxdepth: 3
   :caption: Quick Start Guides
   :hidden:

   Zynq™ Ultrascale+™ <docs/quickstart/mpsoc>
   Versal™ AI Core VCK190 <docs/quickstart/vck190>
 


.. toctree::
   :maxdepth: 3
   :caption: Workflow and Components
   :hidden:

   Overview <docs/workflow>
   DPU IP Details and System Integration <docs/workflow-system-integration>
   Vitis™ AI Model Zoo <docs/workflow-model-zoo>
   Developing a Model for Vitis AI <docs/workflow-model-development>
   Deploying a Model with Vitis AI <docs/workflow-model-deployment>
   
.. toctree::
   :maxdepth: 3
   :caption: Reference Designs
   :hidden:

..   Zynq MPSoC / Kria K26 <ref_design_docs/README_DPUCZDX8G.md>
..   Versal / VCK190 <ref_design_docs/README_DPUCVDX8G.md>

.. toctree::
   :maxdepth: 3
   :caption: Additional Information
   :hidden:

   Vitis™ AI User Guides & IP Product Guides <docs/reference/release_documentation>
   Vitis™ AI Developer Tutorials <https://github.com/Xilinx/Vitis-AI-Tutorials>
   Third-party Inference Stack Integration <docs/workflow-third-party>
   IP and Tools Compatibility <docs/reference/version_compatibility>
   Frequently Asked Questions <docs/reference/faq>
   Branching and Tagging Strategy <docs/install/branching_tagging_strategy>

.. toctree::
   :maxdepth: 3
   :caption: Resources and Support
   :hidden:

   Resources and Support <docs/reference/additional_resources>
   
.. toctree::
   :maxdepth: 3
   :caption: Related AMD Solutions
   :hidden:

   DPU-PYNQ <https://github.com/Xilinx/DPU-PYNQ>
   FINN & Brevitas <https://xilinx.github.io/finn/>
   Inference Server <https://xilinx.github.io/inference-server/>
   Unified Inference Frontend <https://github.com/amd/UIF>   
   Ryzen™ AI Developer Guide ~July 29 <https://ryzenai.docs.amd.com/en/latest/>
   Vitis™ AI ONNX Runtime Execution Provider <https://onnxruntime.ai/docs/execution-providers/community-maintained/Vitis-AI-ExecutionProvider.html>
   Vitis™ Video Analytics SDK <https://xilinx.github.io/VVAS/>
   
   
.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim: