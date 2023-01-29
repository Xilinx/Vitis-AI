..
    Copyright 2022 Xilinx Inc.

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

Xilinx® Vitis™ AI is an integrated development environment that can be leveraged to accelerate AI inference on Xilinx platforms. This toolchain provides optimized IP, tools, libraries, models, as well as resources, such as example designs and tutorials that aid the user throughout the development process.  It is designed with high efficiency and ease-of-use in mind, unleashing the full potential of AI acceleration on Xilinx SoCs and Alveo™ Data Center accelerator cards.

.. figure:: docs/reference/images/VAI_IDE.png
   :width: 1300

   Vitis AI Integrated Development Environment Block Diagram

############################################################
Vitis AI Key Components
############################################################

Vitis AI is composed of the following key components:


* **DPUs** - Configurable computation engines optimized for convolution neural networks. Efficient and scalable IP cores that can be customized to meet the needs of many different applications and devices.
* **Model Zoo**  - A comprehensive set of pre-trained and pre-optimized models that are ready to deploy on Xilinx devices.
* **Model Inspector**  - A tool and methodology through which developers can verify model architecture support.
* **Optimizer** - An optional, commercially licensed tool that enables users to prune a model by up to 90%.
* **Quantizer** - A powerful quantizer that supports model quantization, calibration, and fine tuning.
* **Compiler** - Compiles the quantized model for execution on the target DPU accelerator.
* **Runtime (VART)** - An inference runtime for Embedded applications.
* **Profiler** - Performs an in-depth analysis of the efficiency and utilization of AI inference implementations on the DPU.
* **Library** - Offers high-level C++ APIs for AI applications for embedded and data center use-cases.


.. toctree::
   :maxdepth: 3
   :caption: Vitis AI Introduction
   :hidden:

   Overview <docs/workflow>

.. toctree::
   :maxdepth: 3
   :caption: Release Notes
   :hidden:

   Current Release <docs/reference/release_notes_3.0>

.. toctree::
   :maxdepth: 3
   :caption: Installation
   :hidden:

   System Requirements <docs/reference/system_requirements>
   Host Install Instructions <docs/install/install>
   Target Setup Instructions <docs/board_setup/board_setup>

.. toctree::
   :maxdepth: 3
   :caption: Model Zoo
   :hidden:

   Pre-trained, Optimized Models <docs/workflow-model-zoo>

.. toctree::
   :maxdepth: 3
   :caption: Model Development
   :hidden:

   Developing a NN Model for Vitis AI <docs/workflow-model-development>

.. toctree::
   :maxdepth: 3
   :caption: Model Deployment
   :hidden:

   Deploying a NN Model with Vitis AI <docs/workflow-model-deployment>

.. toctree::
   :maxdepth: 3
   :caption: System Integration
   :hidden:

   Integrating the DPU <docs/workflow-system-integration>

.. toctree::
   :maxdepth: 3
   :caption: Third-Party Tools
   :hidden:

   TVM, TensorFlow Lite, ONNX Runtime <docs/workflow-third-party>

.. toctree::
   :maxdepth: 3
   :caption: Release Documentation
   :hidden:

   Formal Vitis AI Documents <docs/reference/release_documentation>

.. toctree::
   :maxdepth: 3
   :caption: Vitis AI Tutorials
   :hidden:

   Vitis AI Developer Tutorials <https://github.com/Xilinx/Vitis-AI-Tutorials>

.. toctree::
   :maxdepth: 3
   :caption: Related Solutions
   :hidden:

   AMD Inference Server <https://xilinx.github.io/inference-server/>
   Vitis Video Analytics SDK <https://xilinx.github.io/VVAS/>
   FINN & Brevitas <https://xilinx.github.io/finn/>
   DPU-PYNQ <https://github.com/Xilinx/DPU-PYNQ>

.. toctree::
   :maxdepth: 3
   :caption: Resources and Support
   :hidden:

   Resources and Support <docs/reference/additional_resources>

.. toctree::
   :maxdepth: 3
   :caption: FAQ
   :hidden:

   Frequently Asked Questions <docs/reference/faq>
