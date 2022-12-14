======================
Toolchain Integration
======================

.. _vitis-integration:

Vitis Integration
-----------------

The Vitis |trade| workflow specifically targets developers who wish to take a very software-centric approach to Xilinx SoC system development. Vitis AI is differentiated from traditional FPGA flows enabling you to build FPGA acceleration into your applications without the need to develop RTL kernels.

The Vitis workflow enables the integration of the DPU IP as an acceleration kernel that is loaded at runtime in the form of an xclbin
file. To provide developers with a reference platform that can be used as a starting point, the Vitis AI repository includes several `reference designs <https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/tree/vai3.0_update/dpu>`__ for the different DPU architectures and target platforms.

In addition, within the Vitis Tutorials, a tutorial is available which provides the `end-to-end workflow <https://github.com/Xilinx/Vitis-Tutorials/tree/2022.1/Vitis_Platform_Creation/Design_Tutorials/02-Edge-AI-ZCU104>`__ for creating a Vitis Platform for ZCU104 targets.

.. _vivado-integration:

Vivado Integration
------------------

The Vivado |reg| workflow targets traditional FPGA developers. It is important to note that the DPU IP is not currently integrated into the Vivado IP catalog. Currently, in order to update support the latest operators and network topologies at the time of Vitis AI release, the IP is released asynchronously as a `reference design and IP repository <https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/tree/vai3.0_update/dpu>`__.

For more information, refer to the following resources:

-  To integrate the DPU in a Vivado design, see this `tutorial <https://github.com/Xilinx/Vitis-AI-Tutorials/blob/2.0/Tutorials/Vitis-AI-Vivado-TRD/>`__.

-  A quick-start example that assists you in deploying VART on Embedded targets is available `here <https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/blob/vai3.0_update/src/vai_runtime/quick_start_for_embedded.md>`__.

.. _linux-dpu-recipes:

Linux DPU Recipes
-----------------

Yocto and PetaLinux users will require bitbake recipes for the Vitis AI components that are compiled for the target. These recipes are provided in the `source code folder <https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/tree/vai3.0_update/src/petalinux_recipes>`__.

.. important:: For Vitis AI releases >= v2.0, Vivado users (Zynq |reg| Ultrascale+ |trade| and Kria |trade| applications) must compile VART standalone without XRT. However, Vitis users must compile VART with XRT (required for Vitis kernel integration). All designs that leverage Vitis AI require VART, while all Alveo |trade| and Versal |reg| designs must include XRT. By default, the Vitis AI Docker images* `include XRT <https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/tree/vai3.0_update/docker/docker_build_gpu.sh#L40>`__. In addition, the Linux bitbake recipe for VART `assumes <https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/blob/vai3.0_update/src/petalinux_recipes/recipes-vitis-ai/vart/vart_3.0.bb#L17>`__ by default that you are leveraging the Vitis flow. If you are leveraging the DPU in Vivado with Linux, you must either leverage ``vart_3.0_vivado.bb`` or, comment out the line ``PACKAGECONFIG:append = " vitis"`` in the ``vart_3.0.bb`` recipe in order to ensure that you are compiling VART without XRT. Failing to do so will result in runtime errors when executing VART APIs. Specifically, XRT will error out when it attempts to load an xclbin file, a kernel file that is absent in the Vivado flow.

.. _whole-application-acceleration:

Whole Application Acceleration
------------------------------

It is typical in machine learning applications to require some degree of pre-processing such as is illustrated in the example below:

.. figure:: reference/images/waa_preprocess.PNG
   :width: 1300

   Simplified CNN Pre-Processing Pipeline

In addition, many real-world applications for machine learning do not simply employ a single machine learning model. It is very common to cascade multiple object detection networks as a pre-cursor to a final stage (for example, classification, OCR). Throughout this pipeline the meta data must be time-stamped or otherwise attached to the buffer address of the associated frame. Pixels bounded by ROI
(Region-of-Interest) predictions are cropped from the the associated frame. Each of these cropped sub-frame images are then scaled such that the X/Y dimensions of the crop match the input layer dimensions of the downstream network. Some pipelines, such as ReID, will localize, crop, and scale ten or more ROIs from every frame. Each of these crops may require a different scaling factor in order to match the input dimensions of the downstream model in the pipeline. Below is an example:

.. figure:: reference/images/waa_cascade.PNG
   :width: 1300

   Typical Cascaded CNN Pre-Processing Pipeline

These pre-, intermediate, and post-processing operations can significantly impact the overall efficiency of the end-to-end
application. This makes “Whole Application Acceleration” or WAA a very important aspect of Xilinx machine learning solutions. All developers leveraging Xilinx devices for high-performance machine learning applications should learn and understand the benefits of WAA. An excellent starting point for this can be found `here <https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/tree/vai3.0_update/examples/waa>`__.

On a similar vein, if you wish to explore the relevance and capabilites of the `Xilinx Vitis Video Analytics (VVAS)
SDK <https://xilinx.github.io/VVAS/>`__, which while not part of Vitis AI, offers many important features for the development of end-to-end video analytics pipelines that employ multi-stage (cascaded) AI pipelines. VVAS is also applicable to designs that leverage video decoding, transcoding, RTSP streaming, and CMOS sensor interfaces. Another important differentiator of VVAS is that it directly enables software developers to leverage `GStreamer <https://gstreamer.freedesktop.org/>`__ commands to interact with the video pipeline.

Vitis AI Profiler
-----------------

The Vitis AI Profiler is a set of tools that enables you to profile and visualize AI applications based on VART. Because the profiler can be enabled post deployment, there are no code changes required, making it relatively easy to use. Specifically, the Vitis AI Profiler supports profiling and visualization of machine learning pipelines deployed on
Embedded targets with the Vitis AI Runtime. In a typical machine learning pipeline, there are portions of the pipeline that are accelerated on the DPU (DPU subgraph partitions), as well as functions such as pre-processing, and/or custom operators not supported by the DPU. These additional functions may be implemented as a C/C++ kernel, or accelerated using Whole-Application Acceleration or using customized RTL. The Vitis AI Profiler enables the developer to visualize and analyze both system and graph-level performance bottlenecks. Use of the Vitis AI Profiler is a important step for developers who wish to iteratively optimize the entire inference pipeline.

The Vitis AI Profiler is a component of the Vitis AI toolchain installed in the VAI Docker. Source code is not provided.

-  For more information on Vitis AI Profiler see the `Profiling the
   Model <https://docs.xilinx.com/access/sources/dita/topic?isLatest=true&ft:locale=en-US&url=ug1414-vitis-ai&resourceid=kdu1570699882179.html>`__
   section in the Vitis AI User Guide.

-  Examples and additional detail for the Vitis AI Profiler can be found
   `here <https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/tree/vai3.0_update/examples/vai_profiler>`__.

-  A tutorial that provides additional insights on the capabilites of
   the Vitis AI Profiler is available
   `here <https://github.com/Xilinx/Vitis-AI-Tutorials/blob/1.4/Design_Tutorials/16-profiler_introduction/README.md>`__.

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim: