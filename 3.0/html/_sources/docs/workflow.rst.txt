Overview
--------

First Steps
~~~~~~~~~~~

So, you are a new user wondering where to start. In general, there are two primary starting points. Most users will want to start by evaluating the toolchain and running a few examples.  AMD recommends that all users start by downloading and running examples on a supported target platform, and then move on to installation and evaluation of the tools.

The two workflows are as follows:

.. figure:: reference/images/New_User_Flow.PNG

   Vitis AI High-Level New User Workflow
   
.. tip:: In either case, we recommend that you review the entirety of the Vitis AI Github.IO documentation as a first step on your journey with Vitis AI. On these pages, you will find vital information that will augment the PDF User and Product Guides :doc:`../docs/reference/release_documentation`.

In the early stages of evaluation, it is recommended that developers obtain and leverage a supported Vitis AI target platform. Several AMD evaluation platforms are directly supported with pre-built SD card images that enable the developer to evaluate the Vitis AI workflow. Because these images are ready-to-use, there is no immediate need for
the developer to master the integration of the DPU IP. This path provides an excellent starting point for software or data science-centric developers.

If you are not familiar with AMD's Adaptable SoC offerings, you may need better understand the features and performance of AMD Adaptable SoCs before selecting a platform.  Users can review Versal |trade|, Zynq |trade| Ultrascale+ |trade| and Alveo datasheets and documentation, as well as the DPU product guides.  Also important is to review the :doc:`../docs/workflow-model-zoo` performance metrics which will allow you to contrast the relative performance of each target family.  If required, users may also wish to consult with a local FAE or ML Specialist to determine the ideal target product family or device for a given application.


Supported Evaluation Targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vitis |trade| AI 3.0 supports the following targets for evaluation.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Product
     - Supported Devices

   * - Versal Embedded
     - `VCK190 <https://www.xilinx.com/vck190>`__ / `VCK5000 <https://www.xilinx.com/vck5000>`__

   * - Versal AI Edge
     - `VEK280 <https://www.xilinx.com/member/vitis-ai-vek280.html>`__ 

   * - Zynq Ultrascale+ Embedded
     - `ZCU102 <https://www.xilinx.com/zcu102>`__ / `ZCU104 <https://www.xilinx.com/zcu104>`__ / `Kria K26 SOM <https://www.xilinx.com/kria>`__

   * - Alveo Data Center Acceleration Cards
     - `V70 <https://www.xilinx.com/member/v70.html#vitis_ai>`__

Vitis AI support for the `U200 16nm DDR <https://www.xilinx.com/U200>`__, `U250 16 nm DDR <https://www.xilinx.com/U250>`__, `U280 16 nm HBM <https://www.xilinx.com/U280>`__, `U55C 16 nm HBM <https://www.xilinx.com/U55C>`__, `U50 16 nm HBM <https://www.xilinx.com/U50>`__, and `U50LV 16 nm HBM <https://www.xilinx.com/U50LV>`__ has been discontinued. Please leverage a previous release for these targets or contact your local sales team for additional guidance.  

When you are ready to start with one of these pre-built platforms, you should refer to the Quickstart documentation for the respective target. The Quickstart instructions guide users to download a pre-built board image to launch deployment examples that leverage Vitis AI Model Zoo, Vitis AI Library, and Vitis AI Quantizer and Compiler. This is a crucial first step to becoming familiar with Vitis AI.

In addition, developers with access to suitable available hardware platforms can experience pre-built demonstrations available for download through the `Vitis AI Developer page <https://www.xilinx.com/developer/products/vitis-ai.html#demos>`__. Contact your local FAE to arrange a live demonstration.

Last but not least, embedded in the Vitis AI Github repo, there is a folder that in which we may publish demonstrations from time-to-time. You can access the `demos here <https://github.com/Xilinx/Vitis-AI/tree/3.0/demos>`__.


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
