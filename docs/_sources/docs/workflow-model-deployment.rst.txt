Deploying a Model
=================

Workflow for Deploying a Model
------------------------------

Once you have successfully quantized and compiled your model for a specific DPU, the next task is to deploy that model on the target. Follow these steps in this process:

1. Test your model and application software on one of the Xilinx |reg| platforms for which a pre-built DPU image is provided. Ideally, this would be the platform and DPU that closely matches your final production deployment.

2. Customize the Xilinx platform design with any substantive changes required to the DPU IP. Incorporate your final pipeline's pre- or post-processing pipeline acceleration components. Retest your model.

3. Port the Xilinx platform design to your final target hardware platform. Retest your model.

The motivation for this multi-step process is to minimize the number of variables involved in the initial deployment. The process enables the developer to perform verification at each stage. This saves users many hours of frustration and troubleshooting.

In general, the workflow illustrated on the right-hand side of the following diagram is all that is required for the first two steps of deployment. Generally, the platform development component on the left-hand side of the diagram is only required for the third and final step. Part of the convenience of this is that the work can be partitioned between hardware developers and data science teams, enabling the hardware and data science teams to work in parallel, converging at the final step for the deployment on a custom hardware platform. The following image is representative for Vitis designs, but the ability to partition the effort is similar for Vivado designs. Refer to the user documentation for the :ref:`Vivado <vivado-integration>` and :ref:`Vitis <vitis-integration>` workflows for additional details of the hardware platform development workflow.

.. figure:: reference/images/deployment_workflow.PNG
   :width: 1300

   Simplified Vitis Workflow

.. note:: Not captured in this image is the PetaLinux workflow. In the context of Xilinx pre-built target board images, the goal is to enable the developer without requiring that they modify Linux. An important exception to this is for developers who are customizing hardware IPs and peripherals that reside within the memory space of the target CPU/APU and who wish to customize Linux. Also, in some circumstances, it is possible to directly install Vitis AI on the target without rebuilding the kernel image. Refer to :ref:`linux-dpu-recipes` for additional information.

Embedded versus Data Center Workflows
-------------------------------------

The Vitis AI workflow is largely unified for Embedded and Data Center applications but diverges at the deployment stage. There are various reasons for this divergence, including the following:

-  Zynq |reg| Ultrascale+ |trade|, Kria |trade|, and Versal |reg| SoC applications leverage the on-chip processor subsystem (APU) as the host control node for model deployment. Considering optimization and :ref:`whole-application-acceleration` of subgraphs deployed on the SoC APU is crucial.

-  Alveo data center card deployments leverage the AMD64 architecture host for execution of subgraphs that cannot be deployed on the DPU.

-  Zynq Ultrascale+ and Kria designs can leverage the DPU with either the Vivado workflow or the Vitis workflow.

-  Zynq Ultrascale+ and Kria designs built in Vivado do not use XRT.

-  All Vitis designs require the use of XRT.

Vitis AI Library
----------------

The Vitis AI Library provides you with a head-start on model deployment. While it is possible for developers to directly leverage the Vitis AI Runtime APIs to deploy a model on Xilinx platforms, it is often more beneficial to start with a ready-made example that incorporates the various elements of a typical application, including:

-  Simplified CPU-based pre and post-processing implementations.
-  Vitis AI Runtime integration at an application level.

Ultimately most developers will choose one of two paths for production:

-  Directly leverage the VART APIs in their application code.
-  Leverage the VAI Library as a starting point to code their application.

An advantage of leveraging the Vitis AI Library is ease-of-use, while the potential downsides include losing yourself in code that wasn’t intended for your specific use case, and also a lack of recognition on the part of the developer that the pre- and post-processing implementations provided by the Vitis AI Library will not be optimized for :ref:`Whole Application Acceleration <whole-application-acceleration>`.

If you prefer to use the Vitis AI Runtime APIs directly, the code examples provided in the `Vitis AI
Tutorials <https://github.com/Xilinx/Vitis-AI-Tutorials>`__ will offer an excellent starting point.

-  For more information on Vitis AI Libraries, refer to *Vitis AI Library User Guide* (`UG1354 <https://docs.xilinx.com/access/sources/dita/map?isLatest=true&ft:locale=en-US&url=ug1354-xilinx-ai-sdk>`__).
-  The Vitis AI Library quick start guide and open-source is `here <https://github.com/Xilinx/Vitis-AI/tree/v3.0/src/vai_library>`__.

Vitis AI Runtime
----------------

The Vitis AI Runtime (VART) is a set of API functions that support the integration of the DPU into software applications. VART provides a unified high-level runtime for both Data Center and Embedded targets. Key features of the Vitis AI Runtime API are:

-  Asynchronous submission of jobs to the DPU.
-  Asynchronous collection of jobs from the DPU.
-  C++ and Python API implementations.
-  Support for multi-threading and multi-process execution.

For more information on Vitis AI Runtime, refer to the following documentation:

-  For the Vitis AI Runtime API reference, see `VART Programming APIs <https://docs.xilinx.com/access/sources/dita/topic?isLatest=true&ft:locale=en-US&url=ug1414-vitis-ai&resourceid=erl1576053489624.html>`__ and `Deploying and Running the Model <https://docs.xilinx.com/access/sources/dita/topic?isLatest=true&ft:locale=en-US&url=ug1414-vitis-ai&resourceid=zgy1576168058789.html>`__ in the Vitis AI User Guide.

-  A quick-start example to assist you in deploying VART on embedded devices is available `here <https://github.com/Xilinx/Vitis-AI/tree/v3.0/src/vai_runtime/quick_start_for_embedded.md>`__.

-  The Vitis AI Runtime is also provided as `open-source <https://github.com/Xilinx/Vitis-AI/tree/v3.0/src>`__.


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
