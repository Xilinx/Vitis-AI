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

Vitis |trade| AI 3.5 supports the following targets for evaluation.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Product
     - Supported Devices

   * - Versal AI Edge
     - `VEK280 <https://www.xilinx.com/vek280.html>`__ 

   * - Alveo Data Center Acceleration Cards
     - `V70 <https://www.xilinx.com/applications/data-center/v70.html>`__


Vitis |trade| AI 3.5 supports Zynq |trade| Ultrascale+ |trade| and Versal |trade| AI Core architectures, however the IP for these devices is now considered mature and will not be updated with each release. Users should understand that we will continue to support these targets into the future and Vitis AI will update the pre-built board images and reference designs for these architectures with each major release (ie, 4.0).

For minor releases (x.5), users wishing to leverage these architectures have two choices:

#. Leverage Vitis |trade| AI 3.0 for initial evaluation and development.
#. Build a custom board Petalinux image for their target leveraging the Vitis AI 3.5 runtime and libraries.

Vitis AI support for the `VCK5000 <https://www.xilinx.com/VCK5000>`__ was discontinued in the 3.5 release.  Please use Vitis AI 3.0 for this target.

Vitis AI support for the Uxxx Alveo targets was discontinued in the 3.0 release.  Please use Vitis AI 2.5 for those targets.

When you are ready to start with one of these platforms, you should refer to the Quickstart documentation. Quickstart instructions guide users to download a pre-built board image to launch deployment examples that leverage Vitis AI Model Zoo. This is a crucial first step to becoming familiar with Vitis AI.

In addition, developers with access to suitable available hardware platforms can experience pre-built demonstrations available for download through the `Vitis AI Developer page <https://www.xilinx.com/developer/products/vitis-ai.html#demos>`__. Contact your local FAE to arrange a live demonstration.

Last but not least, embedded in the Vitis AI Github repo, there is a folder in which we may publish demonstrations from time-to-time. You can access the `demos here <https://github.com/Xilinx/Vitis-AI/tree/v3.5/demos>`__.


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
