.. 
   Copyright 2021 Xilinx, Inc.

.. meta::
   :keywords: Vitis AI, tutorials, core, development, machine learning, AI, acceleration
   :description: Learn how to use Vitis AI to AI inference on Xilinx hardware platforms, including both edge devices and Alveo cards.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials
 

   

Vitis AI
###########

.. image:: docs/images/Vitis-AI.png

Vitis |copy| AI is a development stack for AI inference on Xilinx|copy| hardware platforms, including both edge devices and Alveo|trade| cards.

It consists of optimized IP, tools, libraries, models, and example designs. It is designed with high efficiency and ease of use in mind, unleashing the full potential of AI acceleration on Xilinx FPGA and ACAP.

.. image:: docs/images/Vitis-AI-arch.png

Vitis AI is composed of the following key components:

* AI Model Zoo - A comprehensive set of pre-optimized models that are ready to deploy on Xilinx devices.
* AI Optimizer - An optional model optimizer that can prune a model by up to 90%. It is separately available with commercial licenses.
* AI Quantizer - A powerful quantizer that supports model quantization, calibration, and fine tuning.
* AI Compiler - Compiles the quantized model to a high-efficient instruction set and data flow.
* AI Profiler - Perform an in-depth analysis of the efficiency and utilization of AI inference implementation.
* AI Library - Offers high-level yet optimized C++ APIs for AI applications from edge to cloud.
* DPU - Efficient and scalable IP cores can be customized to meet the needs for many different applications

For more details on the different DPUs available, refer to `DPU Naming <docs/learn/dpu_naming.md>`_.

Learn More: `Vitis AI Overview <https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html>`_.

.. toctree::
   :maxdepth: 2
   :caption: Learn
   :hidden:

   docs/learn/release_notes
   docs/learn/whats_new
   docs/learn/programming
   docs/learn/system_requirements
   docs/learn/dpu_naming

.. toctree::
   :maxdepth: 2
   :caption: Quick Start
   :hidden:
   
   docs/quick-start/install/README
   docs/quick-start/faq
   docs/quick-start/support
   
.. toctree::
   :maxdepth: 2
   :caption: Get Started
   :hidden:
   
   docs/get-started/examples/README
   docs/get-started/tutorials/README
   
.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:
   
   docs/reference/README
   docs/reference/Thirdpartysource


.. _Versions:

.. toctree::
   :maxdepth: 2
   :caption: Versions
   :hidden:

   Master <https://xilinx.github.io/Vitis-AI/master/docs/README.html>

      
	 


   