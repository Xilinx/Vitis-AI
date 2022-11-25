<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# Vitis AI - A Brief Introduction

The page intends to provide a brief introduction to each component of the Vitis AI workflow, and provide a unified landing page that will assist developers in navigating to key resources for each stage in the workflow.  *We suggest that you review this page in it's entirety as a first step on your journey with Vitis AI.*

## Documentation

Vitis AI documentation consists of both formal product and user guides as well as a selection of task-specific resources and examples.  The formal documentation is listed in the table found [here](#release-documentation).  Additional task-specific resources and examples are encapsulated in the various sections of this introductory page.

## Getting Started Resources

The resource map below can be used to jump directly to the specific tasks and elements of the Vitis AI workflow:

<div id="readme" class="Box-body readme blob js-code-block-container p-5 p-xl-6 gist-border-0">
    <article class="markdown-body entry-content container-lg" itemprop="text"><table>
<tbody><tr>

</th>
</tr>
  <tr>
    <th rowspan="6" width="17%">Getting Started</th>
   </tr>
<tr>
	<td align="center" colspan="3"><a href="#test-drive-vitis-ai-on-a-supported-platform">Test-Drive Vitis AI</a></td>
	<td align="center" colspan="2"><a href="#vitis-ai-installation">Installation</a></td>	
</tr>
  <tr></tr>
<tr>
	<td align="center" colspan="7"><a href="https://github.com/Xilinx/Vitis-AI-Tutorials">Tutorials</a></td>
</tr>
<tr></tr>
    <tr></tr>
  <tr><th colspan="6"></th></tr>
  <tr></tr>
  <tr>
     <th rowspan="7" width="17%">Design and Development</th>
   </tr>
<tr>
	<td align="center"><a href="#model-zoo">Model Zoo</a></td>
	<td align="center"><a href="#model-inspector">Model Inspector</a></td>
	<td align="center"><a href="#operator-support">Supported Operators</a></td>
	<td align="center"><a href="#model-quantization">Quantization/QAT</a></td>
	<td align="center"><a href="#model-compilation">Compilation</a></td>
</tr>
  <tr></tr>
<tr>
	<td align="center"><a href="#model-optimization">Pruning/NAS/OFA</a></td>
	<td align="center"><a href="#model-deployment">Model Deployment</a></td>
	<td align="center"><a href="#vitis-ai-runtime">Vitis AI Runtime</a></td>
	<td align="center"><a href="#vitis-ai-library">Vitis AI Library</a></td>
	<td align="center" colspan="2"><a href="#model-profiling" rel="nofollow">Model Profiling</a></td>
</tr>
<tr>
	<td align="center"><a href="#whole-application-acceleration">Whole Application Acceleration</a></td>
	<td align="center"><a href="#version-compatibility">Vivado Version Compatibility</a></td>
	<td align="center"><a href="#version-compatibility">Vitis Version Compatibility</a></td>
	<td align="center"><a href="#linux-dpu-recipes">Linux DPU Recipes</a></td>
	<td align="center" colspan="2"><a href="#what-is-a-dpu">What is a DPU?</a></td>
</tr>	    
  <tr></tr>
</tbody></table>

With that out of the way, let's get started!

## The Journey for New Users

So, you are a new user and are wondering where to get started?  In general, there are two primary starting points.  Most users will want to start either by installing the toolchain, or doing a "test-drive".  Our recommendation is that all users should start with a "test-drive" and then move on to installation of the tools.  These two work flows are shown below.

<div align="center">
  <img width="100%" height="100%" src="reference/images/New_User_Flow.PNG">
</div>

## What is a DPU?

Before we go much further, it would be useful to understand what is meant by the acronym, D-P-U.  So what is a DPU, exactly?

Xilinx uses this acronym to identify soft accelerators that target deep-learning inference.  These "**D**eep Learning **P**rocessing **U**nits" are a key component of the Vitis-AI solution.  This (perhaps overloaded) term can be used to refer to one of several potential architectures of accelerator, and covering multiple network topologies.

A DPU can be comprised purely of elements that are available in the Xilinx programmable logic fabric, such as DSP, BlockRAM, UltraRAM, LUTs and Flip-Flops, or may be developed as a set of microcoded functions that are deployed on the Xilinx AIE, or "AI Engine" architecture.  Furthermore, in the case of some applications, the DPU is likely to be comprised of both programmable logic and AIE array resources.


## Test-Drive Vitis AI on a Supported Platform
In the early stages of evaluation, it is recommended that developers obtain and leverage a supported Vitis AI target platform.  Several Xilinx evaluation platforms are directly supported with pre-built SD card images that enable the developer to evaluate the Vitis AI workflow.  Because these images are ready-to-use, there is no immediate need for the developer to master the integration of the DPU IP.  This path provides an excellent starting point for developers who are software or data science centric.

To get started, you will need to know which platform you are planning to target.  New users should consult with a local FAE or ML Specialist, review the DPU product guides, review the target platform documentation, and finally, review the [Model Zoo](#model-zoo) performance metrics.


*******************
📎  **Supported evaluation targets:**

#### Versal Embedded: 
  - [VCK190](https://www.xilinx.com/vck190)   /   [VCK5000](https://www.xilinx.com/vck5000)

#### Zynq Ultrascale+ Embedded: 
  - [ZCU102](https://www.xilinx.com/zcu102)   /   [ZCU104](https://www.xilinx.com/zcu104)   /   [Kria K26 SOM](https://www.xilinx.com/kria)

#### Alveo Data Center Acceleration Cards:
  - [U200 16nm DDR](https://www.xilinx.com/U200)   /   [U250 16nm DDR](https://www.xilinx.com/U250)   /   [U280 16nm HBM](https://www.xilinx.com/U280)   /   [U55C 16nm HBM](https://www.xilinx.com/U55C)   /   [U50 16nm HBM](https://www.xilinx.com/U50)   /   [U50LV 16nm HBM](https://www.xilinx.com/U50LV)

  - [Alveo Product Selection Guide](https://www.xilinx.com/content/dam/xilinx/support/documents/selection-guides/alveo-product-selection-guide.pdf)  
*******************


When you are ready to get started with one of these pre-built platforms, you should refer to the target setup instructions [here](../setup).  These instructions walk you as a user through the process of downloading a pre-built board image so that you can launch deployment examples that leverage models from the Vitis AI Model Zoo.  This is a key first step to becoming familiar with Vitis AI.

In addition, developers having access to suitable available hardware platforms can experience pre-built demonstrations that are made are available for download via the [Vitis AI Developer page](https://www.xilinx.com/developer/products/vitis-ai.html#demos).  You can also contact your local FAE to arrange a live demonstration of the same.

Last but not least, embedded in the Vitis AI Github repo, there are a few new demonstrations for NLP and Vision Transformer models and RNN DPU implementations.  You can access the [transformer demos here](../demos/transformer) and the [RNN demos here](../demos/rnn)

Once your "test drive" of the hardware platform is complete, we would recommend that you review the remainder of this introductory page in it's entirety in order to become more familiar with the components of Vitis AI.

## Model Zoo

The Vitis AI Model Zoo is a diverse set of pre-trained models that can be used for reference for development of your own model, or which can potentially be deployed directly in your application.  You can learn more about the Vitis AI Model Zoo [here](../model_zoo).

## Vitis-AI Installation
Vitis-AI is generally installed on a native Linux machine, and in the form of a Docker container.  To install Vitis-AI on your development machine (host machine), visit the installation instructions [here](install/README.md)

##  Model Inspector

-----------------------
📌**Important Note** - The first release of Model Inspector included with Vitis AI 2.5, should be considered to be of "beta" quality.  Please submit Github issues if you encounter problems with this initial release.

-----------------------

The Vitis AI quantizer and compiler are designed to parse and compile operators within a frozen, FP32 graph for acceleration in hardware.  However, novel neural networks architectures, operators and activation types are constantly being developed and optimized for prediction accuracy and performance.  In this context, it is important to understand that while we strive to provide support for a wide variety of neural network architectures and provide these graphs for user reference, not every operator is supported for acceleration on the DPU.  Furthermore, there are specific layer ordering requirements that enable Vitis AI model deployment.

In the early phases of development, it is highly recommended that the developer leverage the Vitis AI Model Inspector as an initial sanity check to confirm that the operators, and sequence of operators in the graph is compatible with Vitis AI.

<div align="center">
  <img width="100%" height="100%" src="reference/images/Model_Inspector_Flow.PNG">
</div>

-----------------------
#### *Model Inspector related resources:*
- When you are ready to get started with the Vitis AI Model Inspector, refer to the examples provided for both [PyTorch](../src/Vitis-AI-Quantizer/vai_q_pytorch/example/inspector_tutorial.ipynb) and [TensorFlow](../src/Vitis-AI-Quantizer/vai_q_tensorflow2.x/#inspecting-vai_q_tensorflow2)
- If you determine that your graph uses operators that are not natively supported by your specific DPU target, there are several paths that you can explore, discussed in [Operator Support](#operator-support)

-----------------------

## Operator Support

In some cases, the user will want to leverage an operator that is not supported for acceleration on the DPU.  There are several obvious paths available, including C/C++ code, or even custom HLS or RTL kernels.  However, these DIY paths pose specific challenges related to the partitioning of a trained model.  For most developers, a workflow that supports automated partitioning is preferred.

-----------------------
📌**Important Note!**  
A high-level list of the supported operators is provided in the DPU IP product [guides](./#release-documentation).  Both the Vitis AI quantizer and compiler implement layer fusion, using a pattern-match algorithm.  The net result of this is that the ordering of layers in the graph can be as important as the operators used.  For instance, if you implement a layer ordering scheme such as CONV -> ReLU -> Batchnorm, the outcome is quite different than [CONV -> Batchnorm -> ReLU](https://support.xilinx.com/s/question/0D52E00006hpW23SAE/resolving-debugging-shiftcut0-tensorflow?language=en_US).  In this context, it is always a very good idea to review the structure of similar Xilinx [Model Zoo](../model_zoo) models to understand how to design your graph for optimum results.

-----------------------

For MPSoC and Versal embedded applications, Xilinx supports an official flow by which the user can add support for these custom operators.  More details can be found [here](../examples/Custom_OP_Demo)

For Alveo cards, the [Whole Graph Optimizer](../examples/WeGO) (WeGO) automatically performs subgraph partitioning for models quantized by Vitis AI quantizer, and applies optimizations and acceleration for the DPU compatible subgraphs. The remaining partitions of the graph are dispatched to the native framework for CPU execution.

In addition, the TVM compiler, TF Lite Delegate, and ONNXruntime Execution Provider (Alveo only) [solutions](../third_party) may also be used to enable support for operations that cannot be accelerated by the DPU.  It is important to note that these third party solutions should be considered to be of "beta" quality and offer more limited support than the standard Vitis AI workflow.


## Model Optimization

The Vitis AI Optimizer exploits the notion of sparsity in order to reduce the overall computational complexity for inference.  Many deep neural networks topologies employ significant levels of redundancy.  This is particularly true when the network backbone was optimized for prediction accuracy with training datasets supporting a large number of classes.  In many cases, this redundancy can be reduced by "pruning" some of the operations out of the graph.  In general, there are two major forms of pruning - channel (kernel) pruning and sparse pruning.

-----------------------
📌**Important Note!**

The Vitis AI Optimizer is an optional tool which can provide considerable uplift in performance in many applications.  However, if your application is not hitting the wall on performance or logic density, or if your model is already well optimized for your dataset and application, you will likely not require the AI Optimizer.

-----------------------

The use of the Vitis AI Optimizer requires that the developer purchase a license for the tool.  You can request a quotation for either the node-locked (part# EF-AI-OPTIMIZER-NL) or floating license (part# EF-AI-OPTIMIZER-FL) by contacting your local [Xilinx Distributor or Sales Office](https://www.xilinx.com/about/contact.html).  **This is a perpetual license with no annual maintenance or renewal costs.**  

Should you wish to evaluate the AI Optimizer prior to considering purchase, you can request access by emailing xilinx_ai_optimizer@amd.com, or request access to the [AI Optimizer Lounge](https://www.xilinx.com/member/ai_optimizer.html)

The Vitis AI Optimizer leverages the native framework in which the model was trained, and both the input and output of the pruning process are a frozen FP32 graph.  At a high-level, the workflow of the AI Optimizer consists of several steps.  The optimizer first performs a sensitivity analysis designed to determine the degree to which each of the convolution kernels (channels) at each layer have an impact on the predictions of the network.  Following this, the kernel weights for channels that are to be pruned are zeroed, permitting accuracy evaluation of the "proposed" pruned model.  The remaining weights are then optimized (fine-tuned) for several training epochs in order to recover accuracy.  Multiple iterations of pruning are typically employed, and after each iteration the state can be captured, permitting the developer to backtrack by one or more pruning iterations.  This ability enables the developer to prune for multiple iterations and then select the iteration with the preferred result.  As necessary, pruning can be restarted from a previous iteration with different hyperparameters in order to address accuracy "cliffs" that may present at a specific iteration.

The final phase of pruning, the transform step, removes the channels that were selected for pruning (previously zeroed weights), resulting in a reduction of the number of channels at each pruned layer in the final computational graph.  For instance, a layer that previously required the computation of 128 channels (128 convolution kernels) may now only require the computation of output activations for 87 channels, (ie, 41 channels were pruned).  Following the transform step, the model is now in a form that can be ingested by the Vitis AI Quantizer and deployed on the target.

The below diagram illustrates the high-level pruning workflow:

<div align="center">
  <img width="100%" height="100%" src="reference/images/optimizer_workflow.PNG">
</div>

###  Channel Pruning

Current Vitis AI DPUs can take advantage of channel pruning in order to greatly reduce the computational cost for inference, often with very little or no loss in prediction accuracy.  In constrast to sparse pruning which requires that the the computation of specific activations within a channel or layer be "skipped" at inference time, channel pruning requires no special hardware to address the problem of these "skipped" computations.

The Vitis AI Optimizer is an optional component of the Vitis AI flow.  In general it is possible to reduce the overall computational cost by a factor of more than 2x, and in some cases by a factor of 10x, with minimal losses in prediction accuracy.  In many cases, there is actually an improvement in prediction accuracy during the first few iterations of pruning.  While the fine-tuning step is in part responsible for this improvement, it is not the only explanation.  Such accuracy improvements will not come as a surprise to developers who are familiar with the concept of overfitting, a phenomena that can occur when a large, deep, network is trained on a dataset that has a limited number of classes.

Many of the pre-trained networks available in the Xilinx [Model Zoo](#model-zoo) were pruned using this technique.

###  Neural Architecture Search

In addition to channel pruning, a technique coined "Once-for-All" training is supported in Vitis-AI.  The concept of Neural Architecture Search (NAS) is that for any given inference task and dataset, there exist in the potential design space a number of network architectures that are both efficient and which have high prediction scores.  Often, a developer starts with a standard backbone that is familiar to them, such as ResNet50, and trains that network for the best accuracy.  However, there are many cases when a network topology with a much lower computational cost may have offered similar or better performance.  For the developer, the effort to train multiple networks with the same dataset (sometimes going so far as to make this a training hyperparameter) is not an efficient method to select the best network topology.  "Once-for-All" addresses this challenge by employing a single training pass and novel selection techniques.

-----------------------
#### *NAS and AI Optimizer related resources:*
- Sample scripts for channel pruning can be found in [examples](../examples/Vitis-AI-Optimizer) 
- For additional details on channel pruning leveraging the Vitis AI Optimizer, please refer to [Vitis AI Optimizer User Guide](#release-documentation).
- For information on Xilinx NAS / Once-for-All, refer to the Once-for-All (OFA) section in the [Vitis AI Optimizer User Guide](#release-documentation)
- Once-for-All examples can be found [here](../examples/ofa)
- An excellent overview of the advantages of OFA is available on the [Xilinx Developer website](https://www.xilinx.com/developer/articles/advantages-of-using-ofa.html)
-----------------------

## Model Quantization

Deployment of neural networks on Xilinx DPUs is made more efficient through the use of integer quantization to reduce the energy cost, memory footprint, and data path bandwidth required for inference.

Xilinx general-purpose CNN-focused DPUs leverage INT8 (8-bit integer) quantization of a trained network.  In many real-world datasets, the distribution of both the weights and activations at a given layer in the network typically spans a much narrower range than can be represented by a 32-bit floating point number.  It is thus possible to accurately represent the distribution of weights and activations at a given layer as integer values by simply applying a scaling factor.  The impact on prediction accuracy of INT8 quantization is typically low, often less than 1%.  This has been found to be true in many applications in which the input data consists of images and video, point-cloud data, as well as input data from a variety of sampled-data systems, including certain audio and RF applications.

###  Quantization Process

The Vitis AI Quantizer, integrated as a component of either TensorFlow or PyTorch, performs a calibration step in which a subset of the original training data (typically 100-1000 samples, no labels required) are forward propagated through the network in order to analyze the distribution of the activations at each layer.  The weights and activations are then quantized as 8-bit integer values.  This process is referred to as Post-Training Quantization.  Following quantization, the prediction accuracy of the network is re-tested using data from the validation set.  If the accuracy is found to be acceptable, the quantization process is complete.

With certain network topologies, the developer may experience excessive accuracy loss.  In these cases a technique referred to as QAT (Quantization Aware Training) can be used with the source training data to execute several back propagation passes with the objective of optimizing (fine-tuning) the quantized weights.


<div align="center">
  <img width="100%" height="100%" src="reference/images/quant_workflow.PNG">
</div>


The Vitis AI Quantizer is a component of the Vitis AI toolchain, installed in the VAI Docker, and is also provided as [open-source](../src).

-----------------------
#### *Quantization related resources:*
- For additional details on the Vitis AI Quantizer, please refer to Chapter 3 "Quantizing the Model" in [the Vitis AI User Guide](#release-documentation).

- TensorFlow 2.x examples are available as follows:</br>
[TF2 Post-Training Quantization](../src/Vitis-AI-Quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/g3doc/guide/quantization/post_training.md)</br>
[TF2 Quantization Aware Training](../src/Vitis-AI-Quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/g3doc/guide/quantization/training.md)

- PyTorch examples are available as follows:</br>
[PT Post-Training Quantization](../src/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_quant.py)</br>
[PT Quantization Aware Training](../src/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_qat.py)
-----------------------

## Model Compilation

Once the model has been quantized, the Vitis AI Compiler is used to construct an internal computation graph as an intermediate representation (IR).  This internal graph consists of independent control and data flow representations. The compiler then performs multiple optimizations, for example, batch normalization operations are fused with convolution when the convolution operator preceeds the normalization operator.  As the DPU supports multiple dimensions of parallelism, efficient instruction scheduling is key to exploiting the inherent parallelism and potential for data reuse in the graph.  Such optimizations are addressed by the Vitis AI Compiler.

The intermediate representation leveraged by Vitis-AI is "XIR" (Xilinx Intermediate Representation).  The XIR-based compiler takes the quantized TensorFlow or PyTorch model as input. First, the compiler transforms the input model into the XIR format. Most of the variations between different frameworks are eliminated at this stage. The compiler then applies optimizations to the graph, and as necessary will partition it into several subgraphs on the basis of whether the subgraph operators can be executed on the DPU. Architecture-aware optimizations are applied for each subgraph. For the DPU subgraph, the compiler generates the instruction stream.  Finally, the optimized graph is serialized into a compiled .xmodel file.

The compilation process leverages an additional input in the form of a DPU arch.json file.  The purpose of this file is to communicate to the compiler the target architecture and hence, capabilities, of the specific DPU for which the graph will be compiled.  If the correct arch.json file is not used, the compiled model will not run on the target.  The implication of this is also that models that were compiled for a specific target DPU must be recompiled if they are to be deployed on a different DPU architecture.  Runtime errors will occur if the model was not compiled for the correct DPU architecture.

Once you have compiled the .xmodel file, you can leverage [Netron](https://github.com/lutzroeder/netron) to review the final graph structure.  Note that as part of the compilation  process the weights are formatted as INT8, concatenated and shuffled in for efficient execution.  Thus, it is not possible to review the weights post-compilation.

The diagram below illustrates a high-level overview of the Vitis AI Compiler workflow:  

<div align="center">
  <img width="100%" height="100%" src="reference/images/compiler_workflow.PNG">
</div>
</br>

The Vitis AI Compiler is a component of the Vitis AI toolchain, installed in the VAI Docker.  The source code for the compiler is not provided.

-----------------------
#### *Compiler related resources:*
- For more information on Vitis AI Compiler and XIR refer to Chapter 4 in the [Vitis AI User Guide](#release-documentation)
- PyXIR, which supports TVM and ONNXRuntime integration is available as [open source](https://github.com/Xilinx/pyxir)
- XIR source code is released as a [component of VART](../src/Vitis-AI-Runtime/VART/xir)
-----------------------

## Model Deployment

Once you have successfully quantized and compiled your model for a specific DPU, the next task is to deploy that model on the target.  Our strong recommendation is that users follow a specific series of steps in this process:

**[Step 1]** *Test your model and application software on one of the Xilinx platforms for which a pre-built DPU image is provided.  Ideally this would be the platform and DPU which most closely matches your final production deployment.*

**[Step 2]** *Customize the Xilinx platform design with any substantive changes required to the DPU IP, and if possible to incorporate the pre/post processing pipeline acceleration components of your final pipeline.  Retest your model.*

**[Step 3]** *Port the Xilinx platform design to your final target hardware platform.  Retest your model.*

The motivation for this multi-step process is simply to minimize the number of variables involved in the initial deployment.  This process enables the developer to perform verification at each stage.  This has been found to save users many hours of frustration and troubleshooting.

In general, workflow illustrated of the right-hand side of the below diagram is all that is required for the initial two steps of deployment.  Generally, the platform development component on the left-hand side of the diagram is only required for the third and final step.  Part of the convenience of this is that the work can be partitioned between hardware developers and data science teams, enabling the hardware and data science teams to work in parallel, converging at the final step for the deployment on a custom hardware platform.  The below image is representative for Vitis designs, but the ability to partition the effort is similar for Vivado designs.  Please refer to the user documentation for the [Vivado](#vivado-integration) and [Vitis](#vitis-integration) workflows for additional details of the hardware platform development workflow.

<div align="center">
  <img width="100%" height="100%" src="reference/images/deployment_workflow.PNG">
</div>

*Not captured in this image is the Petalinux workflow.  In the context of Xilinx pre-built images, the goal is to enable the developer without requiring that they modify Linux.  An important exception to this is for developers who are customizing hardware IP and peripherals that reside within the memory space of the target CPU/APU and who wish to customize Linux.  Also, in some circumstances, it is possible to directly install Vitis AI on the target without rebuilding the kernel image.  Please refer to the [Linux DPU Recipes](#linux-dpu-recipes) section for additional information.*

-----------------------
#### *Deployment related resources:*
- For Alveo developments an [example](../examples/DPUCADF8H) is provided that illustrates how to quantize, compile and deploy models on the U200 and U250
-----------------------


###  Embedded versus Data Center

The Vitis AI workflow is largely unified for Embedded and Data Center applications.  Where the workflow diverges is at the deployment stage.  There are various reasons for this divergence, including the following:

- Zynq Ultrascale+, Kria, and Versal SoC applications leverage the on-chip processor subsystem (APU) as the host control node for model deployment.  It is important to consider optimization and [acceleration](#whole-application-acceleration) of subgraphs that are deployed on the SoC APU
- Alveo deployments leverage the AMD64 architecture host for execution of subgraphs that cannot be deployed on the DPU
- Zynq Ultrascale+ and Kria designs can leverage the DPU with either the Vivado workflow or the Vitis workflow
- Zynq Ultrascale+ and Kria designs built in Vivado do not use XRT
- All Vitis designs require the use of XRT

##  Vitis AI Library

The Vitis AI Library provides users with a head-start on model deployment.  While it is possible for developers to directly leverage the Vitis AI Runtime APIs to deploy a model on Xilinx platforms, it is often more beneficial to start with a ready-made example that incorporates the various elements of a typical application, including:

- Simplified CPU-based pre and post processing implementations
- Vitis AI Runtime integration at an application level

Ultimately most developers will choose one of two paths for production:
- Directly leverage the VART APIs in their application code
- Leverage the VAI Library as a starting point to code their application

An advantage of leveraging the Vitis AI Library is ease-of-use, while the potential downsides include losing yourself in code that wasn't intended for your specific use case, and also a lack of recognition on the part of the developer that the pre and post processing implementations provided by the Vitis AI Library will not be optimized for [Whole Application Acceleration](#whole-application-acceleration).

For users who would prefer to directly use the Vitis AI Runtime APIs, many of the code examples provided in the [Vitis AI Tutorials](https://github.com/Xilinx/Vitis-AI-Tutorials) will offer an excellent starting point.

-----------------------
#### *Vitis AI Library related resources:*
- For more information on Vitis AI Libraries refer to the [Vitis AI Library User Guide](#release-documentation).
- The Vitis AI Library quick start guide as well as open-source can be found [here](../src/Vitis-AI-Library)
-----------------------

##  Vitis AI Runtime

The Vitis AI Runtime (VART) is a set of API functions that support integration of the DPU into software applications.  VART provides a unified high-level runtime for both Data Center and Embedded targets. Key features of the Vitis AI Runtime API are:

- Asynchronous submission of jobs to the DPU
- Asynchronous collection of jobs from the DPU
- C++ and Python API implementations
- Support for multi-threading and multi-process execution

-----------------------
#### *VART related resources:*
- For the Vitis AI Runtime API reference, see Appendix A in the [Vitis AI User Guide](#release-documentation).  Also refer to the section "Deploying and Running the Model" in this same document
- A quick-start example that assists the user in deploying VART on Embedded devices is available [here](../src/Vitis-AI-Runtime/VART/quick_start_for_embedded.md)
- To avoid allowing this to slip into obscurity, we will point out that within VART there are a number of useful [tools](../src/Vitis-AI-Library/usefultools) that any developer who is leveraging VART to deploy on a custom hardware target should be aware of.
- The Vitis AI Runtime is also provided as [open-source](../src).
-----------------------

##  Vitis Integration

The Vitis workflow specifically targets developers who wish to take a very software-centric approach to Xilinx SoC system development.  Vitis AI is differentiated from traditional FPGA flows, enabling the developer to build FPGA acceleration into their applications without the need to develop RTL kernels.

The Vitis workflow enables the integration of the DPU IP as an acceleration kernel that is loaded at runtime in the form of an xclbin file.  To provide developers with a reference platform that can be used as a starting point, the Vitis AI repository includes several [reference designs](../ip_and_reference_designs) for the different DPU architectures and target platforms.

In addition, within the Vitis Tutorials, a tutorial is available which provides the [end-to-end workflow](https://github.com/Xilinx/Vitis-Tutorials/tree/2022.1/Vitis_Platform_Creation/Design_Tutorials/02-Edge-AI-ZCU104) for the creation of a Vitis Platform for ZCU104 targets.

##  Vivado Integration

The Vivado workflow targets traditional FPGA developers.  It is important to note that the DPU IP is not currently integrated into the Vivado IP catalog.  We recognize that this is a little less convenient for developers.  Unfortunately the reasons for this are historic and relate to the independent release scheduling of Vivado, Vitis and Vitis AI.  Currently, in order to update support the latest operators and network topologies at the time of Vitis AI release, the IP is released asynchronously, in the form of a [reference design and IP repository](../ip_and_reference_designs).

-----------------------
#### *Vivado integration related resources:*
- A tutorial is provide that teaches the developer [how to integrate](https://github.com/Xilinx/Vitis-AI-Tutorials/blob/2.5/Design_Tutorials/vivado-integration/README.md) the DPU in a Vivado design
- A quick-start example that assists the user in deploying VART on Embedded targets is available [here](../src/Vitis-AI-Runtime/VART/quick_start_for_embedded.md)
-----------------------

##  Linux DPU Recipes

Yocto and Petalinux users will require bitbake recipes for the Vitis AI components that are compiled for the target.  These recipes, as well as instructions to install Vitis AI as pre-compiled packages, are provided in the [setup folder](../setup/petalinux).

-----------------------
📌**Important Note!**

For Vitis AI releases >= v2.0, Vivado users (Zynq Ultrascale+ and Kria applications) must compile VART standalone without XRT.  However, Vitis users must compile VART with XRT (required for Vitis kernel integration).  All designs that leverage Vitis-AI require VART, while all Alveo and Versal designs must include XRT. By default, the Vitis AI Docker images [include XRT](../docker/docker_build_gpu.sh#L40).  In addition, the Linux bitbake recipe for VART [assumes](../setup/petalinux/recipes-vitis-ai/vart/vart_2.5.bb#L10) by default that the user is leveraging the Vitis flow.  Users who are leveraging the DPU in Vivado with Linux must comment out the line *PACKAGECONFIG:append = " vitis"* in the bitbake recipe in order to ensure that they are compiling VART without XRT.  Failing to do so will result in runtime errors when executing VART APIs.  Specifically, XRT will error out when it attempts to load an xclbin file, a kernel file that will not be present in the Vivado flow.

-----------------------

## Model Profiling

The Vitis AI Profiler is a set of tools that enables the user to profile and visualize AI applications based on VART.  Because the profiler can be enabled post deployment, there are no code changes required, making it relatively easy to use. Specifically, the Vitis AI Profiler supports profiling and visualization of machine learning pipelines deployed on Embedded targets with the Vitis AI Runtime. In a typical machine learning pipeline, there are portions of the pipeline that are accelerated on the DPU (DPU subgraph partitions), as well as functions such as pre-processing, and/or custom operators not supported by the DPU.  These additional functions may be implemented as a C/C++ kernel, or accelerated using Whole-Application Acceleration or using customized RTL.  The Vitis AI Profiler enables the developer to visualize and analyze both system and graph-level performance bottlenecks.  Use of the Vitis AI Profiler is a important step for developers who wish to iteratively optimize the entire inference pipeline.

The Vitis AI Profiler is a component of the Vitis AI toolchain installed in the VAI Docker.  Source code is not provided.

-----------------------
#### *Profiler related resources:*
- For more information on Vitis AI Profiler refer to Chapter 6 in the [Vitis AI User Guide](#release-documentation).
- Examples and additional detail for the Vitis AI Profiler can be found [here](../examples/Vitis-AI-Profiler)
- A tutorial that provides additional insights on the capabilites of the Vitis AI Profiler is available [here](https://github.com/Xilinx/Vitis-AI-Tutorials/blob/1.4/Design_Tutorials/16-profiler_introduction/README.md)
-----------------------

## Whole Application Acceleration

It is typical in machine learning applications to require some degree of pre-processing such as is illustrated in the example below:

<div align="center">
  <img width="100%" height="100%" src="reference/images/waa_preprocess.PNG">
</div>

In addition, many real-world applications for machine learning do not simply employ a single machine learning model.  It is very common to cascade multiple object detection networks as a pre-cursor to a final stage (ie, classification, OCR).  Throughout this pipeline the meta data must be time-stamped or otherwise attached to the buffer address of the associated frame.  Pixels bounded by ROI (Region-of-Interest) predictions are cropped from the the associated frame.  Each of these cropped sub-frame images are then scaled such that the X/Y dimensions of the crop match the input layer dimensions of the downstream network.  Some pipelines, such as ReID, will localize, crop and scale ten or more ROIs from every frame.  Each of these crops may require a different scaling factor in order to match the input dimensions of the downstream model in the pipeline.  Below is an example:

<div align="center">
  <img width="100%" height="100%" src="reference/images/waa_cascade.PNG">
</div>

These pre, intermediate, and post processing operations can significantly impact the overall efficiency of the end-to-end application.  This makes "Whole Application Acceleration" or WAA a very important aspect of Xilinx machine learning solutions.  All developers leveraging Xilinx devices for high-performance machine learning applications should learn and understand the benefits of WAA.  An excellent starting point for this can be found [here](../examples/Whole-App-Acceleration).

On a similar vein, users may wish to explore the relevance and capabilites of the [Xilinx VVAS SDK](https://xilinx.github.io/VVAS/), which while not part of Vitis AI, offers many important features for the development of end-to-end video analytics pipelines that employ multi-stage (cascaded) AI pipelines.  VVAS is also applicable to designs that leverage video decoding, transcoding, RTSP streaming and CMOS sensor interfaces.  Another important differentiator of VVAS is that it directly enables software developers to leverage [GStreamer](https://gstreamer.freedesktop.org/) commands to interact with the video pipeline.

## Version Compatibility

Vitis AI v2.5 and the DPU IP released via the v2.5 branch of this repository are verified compatibile with Vitis, Vivado and Petalinux version 2022.1.  If you are using a previous release of Vitis AI, you should review the [specific version compatibility](reference/version_compatibility.md) for that release. 

## Release Documentation

<div id="readme" class="Box-body readme blob js-code-block-container p-5 p-xl-6 gist-border-0">
    <article class="markdown-body entry-content container-lg" itemprop="text"><table>
<tbody><tr>
</th>
</tr>
  <tr>
    <th rowspan="6" width="17%">Developer User Guides</th>
   </tr>
<tr>
	<td align="center" colspan="6"><a href="https://docs.xilinx.com/r/en-US/ug1414-vitis-ai">Vitis AI User Guide (UG1414)</a></td>
</tr>
     <tr></tr>
<tr>
	<td align="center" colspan="6"><a href="https://docs.xilinx.com/r/en-US/ug1333-ai-optimizer">Optimizer User Guide (UG1333)</a></td>	
</tr>
     <tr></tr>
<tr>
	<td align="center" colspan="6"><a href="https://docs.xilinx.com/r/en-US/ug1354-xilinx-ai-sdk">Library User Guide (UG1354)</a></td>	
</tr>     
     
<tr></tr>
    <tr></tr>
  <tr><th colspan="7"></th></tr>
     <tr></tr>
  <tr>
     <th rowspan="7" width="17%">DPU IP Product Guides</th>
   </tr>
</tr>
<tr>
<tr>
 <td align="center" colspan="1">Zynq Ultrascale+, Kria</a></td>
 <td align="center" colspan="1">Versal Embedded</a></td>
 <td align="center" colspan="4">Alveo</a></td>
 </tr>
 <tr></tr>
 <tr>
 <td align="center" colspan="1">ZCU102, ZCU104, K26 SOM</a></td>
 <td align="center" colspan="1">VCK190</a></td>
 <td align="center" colspan="1">16nm DDR U200, U250</a></td>
 <td align="center" colspan="1">16nm HBM U280, U55C, U50, U50LV</a></td>
 <td align="center" colspan="1">16nm HBM U280, U50LV </a></td>
 <td align="center" colspan="1">Versal VCK5000</a></td>

 </tr>
<tr></tr>
	<td align="center" ><a href="https://docs.xilinx.com/r/en-US/pg338-dpu">DPUCZDX8G (PG338)</a></td>
	<td align="center"><a href="https://docs.xilinx.com/r/en-US/pg389-dpucvdx8g">DPUCVDX8G (PG389)</a></td>
	<td align="center"><a href="https://docs.xilinx.com/r/en-US/pg400-dpucadf8h">DPUCADF8H (PG400)</a></td>
	<td align="center"><a href="https://docs.xilinx.com/r/en-US/pg367-dpucahx8h">DPUCAHX8H (PG367)</a></td>
	<td align="center"><a href="https://docs.xilinx.com/r/en-US/pg366-dpucahx8l">DPUCAHX8L (PG366)</a></td>
	<td align="center"><a href="https://docs.xilinx.com/r/en-US/pg403-dpucvdx8h">DPUCVDX8H (PG403)</a></td>
 <tr><th colspan="7"></th></tr>

 </tbody></table>


## What's Next? 

If you have taken an initial "test drive" and reviewed this brief introductory page we would suggest that you check out one or more of the [tutorials](https://github.com/Xilinx/Vitis-AI-Tutorials) intended to show you how to deploy your deep learning model!

