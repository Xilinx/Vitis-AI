<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# Vitis AI - A Brief Introduction

The page intends to provide a brief introduction to each component of the Vitis AI workflow, and provide a unified landing page that will assist developers in navigating to key resources for each stage in the workflow.  *We suggest that you review this page in it's entirety as a first step on your journey with Vitis AI.*

## Documentation

Vitis AI documentation consists of both formal product and user guides as well as a selection of task-specific resources and examples.  The formal documentation is listed in the table found [here](reference/release_documentation).  Additional task-specific resources and examples are encapsulated in the various sections of this documentation repository.

## Getting Started Resources
The resource map below (markdown only) or sidebar table-of-contents (HTML / Github.IO only) can be used to jump directly to the specific tasks and elements of the Vitis AI workflow.

<div id="readme" class="Box-body readme blob js-code-block-container p-5 p-xl-6 gist-border-0">
    <article class="markdown-body entry-content container-lg" itemprop="text"><table>
﻿<table class="sphinxhide">
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

## The Journey for New Users

So, you are a new user and are wondering where to get started?  In general, there are two primary starting points.  Most users will want to start either by installing the toolchain, or doing a "test-drive".  Our recommendation is that all users should start with a "test-drive" and then move on to installation of the tools.  These two work flows are shown below.

<div align="center">
  <img width="100%" height="100%" src="reference/images/New_User_Flow.PNG">
</div>
</br>

## What is a DPU?

Before we go much further, it would be useful to understand what is meant by the acronym, D-P-U.  So what is a DPU, exactly?

Xilinx uses this acronym to identify soft accelerators that target deep-learning inference.  These "**D**eep Learning **P**rocessing **U**nits" are a key component of the Vitis-AI solution.  This (perhaps overloaded) term can be used to refer to one of several potential architectures of accelerator, and covering multiple network topologies.

A DPU can be comprised purely of elements that are available in the Xilinx programmable logic fabric, such as DSP, BlockRAM, UltraRAM, LUTs and Flip-Flops, or may be developed as a set of microcoded functions that are deployed on the Xilinx AIE, or "AI Engine" architecture.  Furthermore, in the case of some applications, the DPU is likely to be comprised of both programmable logic and AIE array resources.

Each DPU architecture has it's own instruction set, and the Vitis AI Compiler targets that instruction set with the neural network operators to be deployed in the source network.

An example of the DPUCZ, targeting Zynq Ultrascale+ devices is shown in the image below:

<div align="center">
  <img width="100%" height="100%" src="reference/images/DPUCZ.PNG">
</div>
</br>

Vitis AI provides both the DPU IP as well as the required tools to deploy both standard and custom neural networks on Xilinx targets:


<div align="center">
  <img width="100%" height="100%" src="reference/images/VAI-1000ft.PNG">
</div>
</br>


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


When you are ready to get started with one of these pre-built platforms, you should refer to the target setup instructions [here](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/docs/board_setup).  These instructions walk you as a user through the process of downloading a pre-built board image so that you can launch deployment examples that leverage models from the Vitis AI Model Zoo.  This is a key first step to becoming familiar with Vitis AI.

In addition, developers having access to suitable available hardware platforms can experience pre-built demonstrations that are made are available for download via the [Vitis AI Developer page](https://www.xilinx.com/developer/products/vitis-ai.html#demos).  You can also contact your local FAE to arrange a live demonstration of the same.

Last but not least, embedded in the Vitis AI Github repo, there are a few new demonstrations for NLP and Vision Transformer models and RNN DPU implementations.  You can access the [transformer demos here](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/demos/transformer) and the [RNN demos here](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/demos/rnn)

Once your "test drive" of the hardware platform is complete, we would recommend that you review the remainder of this documentation in it's entirety in order to become more familiar with the components of Vitis AI.


## Version Compatibility

Vitis AI v2.5 and the DPU IP released via the v2.5 branch of this repository are verified compatibile with Vitis, Vivado and Petalinux version 2022.1.  If you are using a previous release of Vitis AI, you should review the [specific version compatibility](reference/version_compatibility) for that release. 


