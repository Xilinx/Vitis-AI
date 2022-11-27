##  Vitis Integration

The Vitis workflow specifically targets developers who wish to take a very software-centric approach to Xilinx SoC system development.  Vitis AI is differentiated from traditional FPGA flows, enabling the developer to build FPGA acceleration into their applications without the need to develop RTL kernels.

The Vitis workflow enables the integration of the DPU IP as an acceleration kernel that is loaded at runtime in the form of an xclbin file.  To provide developers with a reference platform that can be used as a starting point, the Vitis AI repository includes several [reference designs](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/dpu) for the different DPU architectures and target platforms.

In addition, within the Vitis Tutorials, a tutorial is available which provides the [end-to-end workflow](https://github.com/Xilinx/Vitis-Tutorials/tree/2022.1/Vitis_Platform_Creation/Design_Tutorials/02-Edge-AI-ZCU104) for the creation of a Vitis Platform for ZCU104 targets.

##  Vivado Integration

The Vivado workflow targets traditional FPGA developers.  It is important to note that the DPU IP is not currently integrated into the Vivado IP catalog.  We recognize that this is a little less convenient for developers.  Unfortunately the reasons for this are historic and relate to the independent release scheduling of Vivado, Vitis and Vitis AI.  Currently, in order to update support the latest operators and network topologies at the time of Vitis AI release, the IP is released asynchronously, in the form of a [reference design and IP repository](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/dpu).

-----------------------
#### *Vivado integration related resources:*
- A tutorial is provide that teaches the developer [how to integrate](https://gitenterprise.xilinx.com/quentonh/Vitis-AI-Tutorials/tree/2.5/Tutorials/DPU_IP_Vivado_Vitis_AI) the DPU in a Vivado design
- A quick-start example that assists the user in deploying VART on Embedded targets is available [here](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src/Vitis-AI-Runtime/VART/quick_start_for_embedded.md)
-----------------------

##  Linux DPU Recipes

Yocto and Petalinux users will require bitbake recipes for the Vitis AI components that are compiled for the target.  These recipes are provided in the [source code folder](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src/petalinux_recipes).

-----------------------
📌**Important Note!** - For Vitis AI releases >= v2.0, Vivado users (Zynq Ultrascale+ and Kria applications) must compile VART standalone without XRT.  However, Vitis users must compile VART with XRT (required for Vitis kernel integration).  All designs that leverage Vitis-AI require VART, while all Alveo and Versal designs must include XRT. By default, the Vitis AI Docker images [include XRT](../docker/docker_build_gpu.sh#L40).  In addition, the Linux bitbake recipe for VART [assumes](../setup/petalinux/recipes-vitis-ai/vart/vart_2.5.bb#L10) by default that the user is leveraging the Vitis flow.  Users who are leveraging the DPU in Vivado with Linux must comment out the line *PACKAGECONFIG:append = " vitis"* in the bitbake recipe in order to ensure that they are compiling VART without XRT.  Failing to do so will result in runtime errors when executing VART APIs.  Specifically, XRT will error out when it attempts to load an xclbin file, a kernel file that will not be present in the Vivado flow.

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
