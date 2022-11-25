<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

![Release Version](https://img.shields.io/github/v/release/Xilinx/Vitis-AI)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr-raw/Xilinx/Vitis-AI)
[![Documentation](https://img.shields.io/badge/documentation-github.IO-blue.svg)](https://xilinx.github.io/Vitis-AI/)


<br />
Xilinx&reg; Vitis&trade; AI is an Integrated Development Environment that can be leveraged to accelerate AI inference on Xilinx platforms. Vitis AI provides optimized IP, tools, libraries, models, as well as resources, such as example designs and tutorials that aid the user throughout the development process.  It is designed with high efficiency and ease-of-use in mind, unleashing the full potential of AI acceleration on Xilinx SoCs and Alveo Data Center accelerator cards.  
<br /> <br />


<div align="center">
  <img width="100%" height="100%" src="docs/reference/images/VAI_IDE.PNG">
</div>
<br />
Vitis AI is composed of the following key components:
<br /> <br />

* **DPUs** - Configurable computation engines optimized for convolution neural networks. Efficient and scalable IP cores that can be customized to meet the needs of many different applications and devices
* **VAI Model Zoo**  - A comprehensive set of pre-trained and pre-optimized models that are ready to deploy on Xilinx devices
* **VAI Model Inspector**  - A tool and methodology through which developers can verify model architecture support
* [**VAI Optimizer**](https://docs.xilinx.com/r/en-US/ug1333-ai-optimizer) - An optional, commercially licensed tool that enables users to prune a model by up to 90%
* [**VAI Quantizer**](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Vitis-AI-Quantizer) - A powerful quantizer that supports model quantization, calibration, and fine tuning
* [**VAI Compiler**](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Vitis-AI-Compiler) - Compiles the quantized model for execution on the target DPU accelerator
* [**VAI Runtime (VART)**](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Vitis-AI-Runtime) - An inference runtime for Embedded applications
* [**VAI Profiler**](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Vitis-AI-Profiler) - Performs an in-depth analysis of the efficiency and utilization of AI inference implementations on the DPU
* [**VAI Library**](https://docs.xilinx.com/r/en-US/ug1354-xilinx-ai-sdk) - Offers high-level C++ APIs for AI applications for embedded and data center use-cases
<br /> <br />

In addition, Vitis AI incorporates acceleration support for several commonly used inference frameworks:
<br /> <br />

* **ONNXRuntime**  - A runtime framework that enables the Xilinx DPU as an "Execution Provider" for ONNX model acceleration
* **TensorFlow Lite** - A workflow that enables integration of the Xilinx DPU as a "Delegate" for model acceleration
* **TVM Inference Compiler** - An open-source inference compiler that enables the Xilinx DPU, and supports a wide variety of machine learning frameworks
<br /> <br />

## Getting Started

If your visit here is accidental, but you are enthusiastic to learn more about Vitis AI, please visit the Vitis AI [homepage](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html) on Xilinx.com

Otherwise, if your visit is deliberate and you are ready to begin, may we suggest one of the following starting points?
- [Visit](https://xilinx.github.io/Vitis-AI/) the Vitis AI documentation pages on Github.IO
- [Jump](docs) to the introductory page, a comprehensive set of resources for both new and experienced users
- [Test drive](docs#test-drive-vitis-ai-on-a-supported-platform) pre-built board images on a supported hardware platform
- [Review](model_zoo) the Vitis AI Model Zoo Performance & Accuracy Data
- [Install](docs/install) Vitis AI
- [Check](https://github.com/Xilinx/Vitis-AI-Tutorials) out tutorials intended to show you how to deploy your deep learning model
- [Leverage](https://xilinx.github.io/inference-server/) the Xilinx Inference Server to deploy inference applications via HTTP REST and WebSocket protocols using an API based on KServe's v2 specification
- [Access](https://www.xilinx.com/developer/products/vitis-ai.html) training, tutorial videos, and more on the Vitis-AI developer pages


## Release Notes & Tool Version Compatibility  
Please see version release notes and tool compatibility information [here](docs/reference/release_notes.md)

## How to Download the Repository

To get a local copy of Vitis AI, clone this repository to the local system with the following command:

```
git clone https://github.com/Xilinx/Vitis-AI
```

This command needs to be executed only once to retrieve the latest version of Vitis AI.

Optionally, configure git-lfs in order to reduce the local storage requirements.



## Questions and Support
- [FAQs](docs/reference/faq.md)
- [Vitis AI Forum](https://support.xilinx.com/s/topic/0TO2E000000YKY9WAO/vitis-ai-ai)
- [Vitis-AI Github branching and tagging strategy](docs/install/branching_tagging_strategy.md)


## Licenses

Vitis AI License: [Apache 2.0](LICENSE)</br>
Third party: [Components](/docs/reference/Thirdpartysource.md)

