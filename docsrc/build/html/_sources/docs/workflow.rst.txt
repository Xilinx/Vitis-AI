===============================
Vitis AI - A Brief Introduction
===============================

The page intends to provide a brief introduction to each component of the Vitis AI |trade| workflow, and provide a unified landing page that will assist developers in navigating to key resources for each stage in the workflow. 

.. tip:: Review this page in its entirety as a first step on your journey with Vitis AI.

Documentation
-------------

Vitis AI documentation consists of both formal product and user guides as well as a selection of task-specific resources and examples. The formal documentation is listed in :doc:`../docs/reference/release_documentation`. Additional task-specific resources and examples are encapsulated in the various sections of this documentation repository.

Getting Started Resources
-------------------------

Use the sidebar ToC to jump directly to the specific tasks and elements of the Vitis AI workflow.

The Journey for New Users
-------------------------

So, you are a new user and are wondering where to get started? In general, there are two primary starting points. Most users will want to start either by installing the toolchain, or doing a “test-drive”. Xilinx recommends that all users should start with a “test-drive” and then move on to installation of the tools. These two workflows are shown below.

.. image:: reference/images/New_User_Flow.PNG

What is a DPU?
--------------

Before you go much further, it would be useful to understand what is meant by the acronym, D-P-U. So what is a DPU, exactly?

Xilinx uses this acronym to identify soft accelerators that target deep-learning inference. These “**D** eep Learning **P** rocessing **U** nits” are a key component of the Vitis-AI solution. This (perhaps overloaded) term can be used to refer to one of several potential architectures of accelerator, and covering multiple network topologies.

A DPU can be comprised purely of elements that are available in the Xilinx |reg| programmable logic fabric, such as DSP, BlockRAM, UltraRAM, LUTs and Flip-Flops, or may be developed as a set of microcoded functions that are deployed on the Xilinx AI Engine, or “AI Engine” architecture. Furthermore, in the case of some applications, the DPU is likely to be comprised of both programmable logic and AI Engine array resources.

Each DPU architecture has its own instruction set, and the Vitis AI Compiler targets that instruction set with the neural network operators to be deployed in the source network.

An example of the DPUCZ, targeting Zynq |reg| Ultrascale+ |trade| devices is shown in the following image:

.. image:: reference/images/DPUCZ.PNG

Vitis AI provides both the DPU IP as well as the required tools to deploy both standard and custom neural networks on Xilinx targets:

.. image:: reference/images/VAI-1000ft.PNG

Test-Drive Vitis AI on a Supported Platform
--------------------------------------------

In the early stages of evaluation, it is recommended that developers obtain and leverage a supported Vitis AI target platform. Several Xilinx evaluation platforms are directly supported with pre-built SD card images that enable the developer to evaluate the Vitis AI workflow. Because these images are ready-to-use, there is no immediate need for
the developer to master the integration of the DPU IP. This path provides an excellent starting point for developers who are software or data science centric.

To get started, you will need to know which platform you are planning to target. New users should consult with a local FAE or ML Specialist, review the DPU product guides, review the target platform documentation, and finally, review the :doc:`workflow-model-zoo` performance metrics.

Supported Evaluation Targets
----------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Product
     - Supported Devices

   * - Versal Embedded
     - `VCK190 <https://www.xilinx.com/vck190>`__ / `VCK5000 <https://www.xilinx.com/vck5000>`__
	 
   * - Versal AI Edge
     - `VEK280 <https://www.xilinx.com/member/dpu-vek280.html>`__ 

   * - Zynq Ultrascale+ Embedded
     - `ZCU102 <https://www.xilinx.com/zcu102>`__ / `ZCU104 <https://www.xilinx.com/zcu104>`__ / `Kria K26 SOM <https://www.xilinx.com/kria>`__

   * - Alveo Data Center Acceleration Cards
     - `U200 16nm DDR <https://www.xilinx.com/U200>`__ / `U250 16 nm DDR <https://www.xilinx.com/U250>`__ / `U280 16 nm HBM <https://www.xilinx.com/U280>`__ / `U55C 16 nm HBM <https://www.xilinx.com/U55C>`__ / `U50 16 nm HBM <https://www.xilinx.com/U50>`__ / `U50LV 16 nm HBM <https://www.xilinx.com/U50LV>`__ / `V70 <https://www.xilinx.com/member/dpu-v70.html>`__

.. note:: See the `Alveo Product Selection Guide <https://www.xilinx.com/content/dam/xilinx/support/documents/selection-guides/alveo-product-selection-guide.pdf>`__ for more information on device selection.

When you are ready to get started with one of these pre-built platforms, you should refer to the :doc:`Target Setup Instructions <board_setup/board_setup>`.
These instructions walk you as a user through the process of downloading a pre-built board image so that you can launch deployment examples that leverage models from the Vitis AI Model Zoo. This is a key first step to becoming familiar with Vitis AI.

In addition, developers having access to suitable available hardware platforms can experience pre-built demonstrations that are made are available for download via the `Vitis AI Developer page <https://www.xilinx.com/developer/products/vitis-ai.html#demos>`__. You can also contact your local FAE to arrange a live demonstration of the same.

Last but not least, embedded in the Vitis AI Github repo, there are a few new demonstrations for NLP and Vision Transformer models and RNN DPU implementations. You can access the `transformer demos here <https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/tree/vai3.0_update/demos/transformer>`__ and the `RNN demos here <https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/tree/vai3.0_update/demos/rnn>`__

Once your “test drive” of the hardware platform is complete, we would recommend that you review the remainder of this documentation in its entirety in order to become more familiar with the components of Vitis AI.

Version Compatibility
---------------------

Vitis AI v3.0 and the DPU IP released via the v3.0 branch of this repository are verified compatibile with Vitis, Vivado and PetaLinux version 2022.2. If you are using a previous release of Vitis AI, you should review the :doc:`../docs/reference/version_compatibility` for that release.


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:

