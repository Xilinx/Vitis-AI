<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

****************
:pushpin:**Important Note!** In most cases, Developers will want to leverage the GPU-enabled Docker as it provides support for accelerated quantization.  Prior to installing Docker, please ensure that you understand the Nvidia driver, CUDA [System Requirements](../reference/system_requirements.md) for Vitis AI.
****************


## Installing NVIDIA Container Toolkit

If you are building the Vitis AI Docker Image with GPU acceleration, you must install the NVIDIA Container Toolkit which enables GPU support inside the Docker container.  Please refer to the official NVIDIA [documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for additional information.

For Ubuntu distributions, Nvidia driver and Container Toolkit installation can generally be accomplished as shown in the following example (use sudo for non-root users):

```
apt purge nvidia* libnvidia*
apt install nvidia-driver-xxx
apt install nvidia-container-toolkit
```
Where xxx is the version of driver that you are choosing to install (ie, *nvidia-driver-510*), and is a version that meets Vitis AI [System Requirements](../reference/system_requirements.md).

A simple test to confirm driver installation is to run nvidia-smi:

```
nvidia-smi
```
The output should appear similar to this:

```
-Need to fill this in-
```
Users should reference [Nvidia driver installation](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) for further details of driver installation.

## Installing Docker

Once you are confident that your system meets any pre-requisites for Vitis AI Docker GPU support, please refer to official Docker [documentation](https://docs.docker.com/engine/install/) to install the Docker engine.

Click here to return to the Vitis AI installation documentation [page](../#installation-steps).
