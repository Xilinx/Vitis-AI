
## Model Deployment

Once you have successfully quantized and compiled your model for a specific DPU, the next task is to deploy that model on the target.  Our strong recommendation is that users follow a specific series of steps in this process:

**[Step 1]** *Test your model and application software on one of the Xilinx platforms for which a pre-built DPU image is provided.  Ideally this would be the platform and DPU which most closely matches your final production deployment.*

**[Step 2]** *Customize the Xilinx platform design with any substantive changes required to the DPU IP, and if possible to incorporate the pre/post processing pipeline acceleration components of your final pipeline.  Retest your model.*

**[Step 3]** *Port the Xilinx platform design to your final target hardware platform.  Retest your model.*

The motivation for this multi-step process is simply to minimize the number of variables involved in the initial deployment.  This process enables the developer to perform verification at each stage.  This has been found to save users many hours of frustration and troubleshooting.

In general, workflow illustrated of the right-hand side of the below diagram is all that is required for the initial two steps of deployment.  Generally, the platform development component on the left-hand side of the diagram is only required for the third and final step.  Part of the convenience of this is that the work can be partitioned between hardware developers and data science teams, enabling the hardware and data science teams to work in parallel, converging at the final step for the deployment on a custom hardware platform.  The below image is representative for Vitis designs, but the ability to partition the effort is similar for Vivado designs.  Please refer to the user documentation for the [Vivado](workflow-system-integration.html:vivado-integration) and [Vitis](workflow-system-integration:#vitis-integration) workflows for additional details of the hardware platform development workflow.

<div align="center">
  <img width="100%" height="100%" src="reference/images/deployment_workflow.PNG">
</div>
</br>

*Not captured in this image is the Petalinux workflow.  In the context of Xilinx pre-built images, the goal is to enable the developer without requiring that they modify Linux.  An important exception to this is for developers who are customizing hardware IP and peripherals that reside within the memory space of the target CPU/APU and who wish to customize Linux.  Also, in some circumstances, it is possible to directly install Vitis AI on the target without rebuilding the kernel image.  Please refer to the [Linux DPU Recipes](workflow-system-integration#linux-dpu-recipes) section for additional information.*

-----------------------
#### *Deployment related resources:*
- For Alveo developments an [example](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/examples/alveo) is provided that illustrates how to quantize, compile and deploy models on the U200 and U250
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
- For more information on Vitis AI Libraries refer to the [Vitis AI Library User Guide](reference/release_documentation).
- The Vitis AI Library quick start guide as well as open-source can be found [here](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src)
-----------------------

##  Vitis AI Runtime

The Vitis AI Runtime (VART) is a set of API functions that support integration of the DPU into software applications.  VART provides a unified high-level runtime for both Data Center and Embedded targets. Key features of the Vitis AI Runtime API are:

- Asynchronous submission of jobs to the DPU
- Asynchronous collection of jobs from the DPU
- C++ and Python API implementations
- Support for multi-threading and multi-process execution

-----------------------
#### *VART related resources:*
- For the Vitis AI Runtime API reference, see Appendix A in the [Vitis AI User Guide](reference/release_documentation).  Also refer to the section "Deploying and Running the Model" in this same document
- A quick-start example that assists the user in deploying VART on Embedded devices is available [here](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src/Vitis-AI-Runtime/VART/quick_start_for_embedded.md)
- To avoid allowing this to slip into obscurity, we will point out that within VART there are a number of useful [tools](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src/Vitis-AI-Library/usefultools) that any developer who is leveraging VART to deploy on a custom hardware target should be aware of.
- The Vitis AI Runtime is also provided as [open-source](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src).
-----------------------


## Whole Application Acceleration

It is typical in machine learning applications to require some degree of pre-processing such as is illustrated in the example below:

<div align="center">
  <img width="100%" height="100%" src="reference/images/waa_preprocess.PNG">
</div>
</br>

In addition, many real-world applications for machine learning do not simply employ a single machine learning model.  It is very common to cascade multiple object detection networks as a pre-cursor to a final stage (ie, classification, OCR).  Throughout this pipeline the meta data must be time-stamped or otherwise attached to the buffer address of the associated frame.  Pixels bounded by ROI (Region-of-Interest) predictions are cropped from the the associated frame.  Each of these cropped sub-frame images are then scaled such that the X/Y dimensions of the crop match the input layer dimensions of the downstream network.  Some pipelines, such as ReID, will localize, crop and scale ten or more ROIs from every frame.  Each of these crops may require a different scaling factor in order to match the input dimensions of the downstream model in the pipeline.  Below is an example:

<div align="center">
  <img width="100%" height="100%" src="reference/images/waa_cascade.PNG">
</div>
</br>

These pre, intermediate, and post processing operations can significantly impact the overall efficiency of the end-to-end application.  This makes "Whole Application Acceleration" or WAA a very important aspect of Xilinx machine learning solutions.  All developers leveraging Xilinx devices for high-performance machine learning applications should learn and understand the benefits of WAA.  An excellent starting point for this can be found [here](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/examples/waa).

On a similar vein, users may wish to explore the relevance and capabilites of the [Xilinx VVAS SDK](https://xilinx.github.io/VVAS/), which while not part of Vitis AI, offers many important features for the development of end-to-end video analytics pipelines that employ multi-stage (cascaded) AI pipelines.  VVAS is also applicable to designs that leverage video decoding, transcoding, RTSP streaming and CMOS sensor interfaces.  Another important differentiator of VVAS is that it directly enables software developers to leverage [GStreamer](https://gstreamer.freedesktop.org/) commands to interact with the video pipeline.


## Model Profiling

The Vitis AI Profiler is a set of tools that enables the user to profile and visualize AI applications based on VART.  Because the profiler can be enabled post deployment, there are no code changes required, making it relatively easy to use. Specifically, the Vitis AI Profiler supports profiling and visualization of machine learning pipelines deployed on Embedded targets with the Vitis AI Runtime. In a typical machine learning pipeline, there are portions of the pipeline that are accelerated on the DPU (DPU subgraph partitions), as well as functions such as pre-processing, and/or custom operators not supported by the DPU.  These additional functions may be implemented as a C/C++ kernel, or accelerated using Whole-Application Acceleration or using customized RTL.  The Vitis AI Profiler enables the developer to visualize and analyze both system and graph-level performance bottlenecks.  Use of the Vitis AI Profiler is a important step for developers who wish to iteratively optimize the entire inference pipeline.

The Vitis AI Profiler is a component of the Vitis AI toolchain installed in the VAI Docker.  Source code is not provided.

-----------------------
#### *Profiler related resources:*
- For more information on Vitis AI Profiler refer to Chapter 6 in the [Vitis AI User Guide](reference/release_documentation).
- Examples and additional detail for the Vitis AI Profiler can be found [here](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/examples/Vitis-AI-Profiler)
- A tutorial that provides additional insights on the capabilites of the Vitis AI Profiler is available [here](https://github.com/Xilinx/Vitis-AI-Tutorials/blob/1.4/Design_Tutorials/16-profiler_introduction/README.md)
-----------------------