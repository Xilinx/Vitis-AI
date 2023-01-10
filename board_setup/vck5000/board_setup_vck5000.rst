=====================================
Setting up a Versal Accelerator Card
=====================================

The Xilinx |reg| DPUs for VCK5000-PROD card is a High Performance CNN processing engine **DPUCVDX8H**. The detailed combination of VCK5000-PROD card and DPU IP  is shown in the following table, you can choose one of them according to your own situation.

=== ================ =====================
No. Accelerator Card DPU IP
=== ================ =====================
1   VCK5000-PROD     DPUCVDX8H_4pe_miscdwc
2   VCK5000-PROD     DPUCVDX8H_6pe_dwc
3   VCK5000-PROD     DPUCVDX8H_6pe_misc
4   VCK5000-PROD     DPUCVDX8H_8pe_normal
=== ================ =====================

1. VCK5000-PROD Card Setup in Host
------------------------------

We provide some scripts to help to automatically finish the VCK5000-PROD card setup process. You could refer to these to understand the required steps. To use the scripts, just input the command below. It will detect Operating System you are using, then download and install the appropriate packages. Suppose you have downloaded Vitis-AI, entered Vitis-AI directory.

   .. note:: You should use this script in host environment, namely out of the Docker container. After the script is executed successfully, manually reboot the host server once. For cloud DPU, Vitis AI 3.0 applies 2022.2 Tools/Platform/XRT/XRM.

   ::

      source ./install.sh

The following installation steps were performed in this script.

   - Install XRT.
   - Install XRM. The `Xilinx Resource Manager (XRM) <https://github.com/Xilinx/XRM/>`__ manages and controls FPGA resources on a machine. It is used by the runtime.
   - Install the VCK5000-PROD Card Target Platform.
   - Install DPU V4E xclbin for VCK5000-PROD.

After the script is executed successfully, use the XRT command to check that the installation was successful. The result should contain the correct information for System Configuration, XRT and Devices present.

   ::

      /opt/xilinx/xrt/bin/xbutil examine


   .. note:: This version requires the use of a VCK5000-PROD card. VCK5000-ES1 card is no longer updated since Vitis AI 2.0, if you want to use it, refer to `Vitis AI 1.4.1 <https://github.com/Xilinx/Vitis-AI/tree/v1.4.1>`__.

2. Environment Variable Setup in Docker Container
-------------------------------------------------

Suppose you have downloaded Vitis-AI, entered Vitis-AI directory, and then started Docker image. In the docker container, execute the following steps. You can use the following command to set environment variables. It should be noted that the xclbin file should be in the
``/opt/xilinx/overlaybins`` directory. There are four xclbins to choose from depending on the parameters you use.

   - For 4PE 350 Hz, you can select DPU IP via the following command.

   ::

      source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_4pe_miscdwc

   - For 6PE 350 Hz with DWC, you can select DPU IP via the following command.

   ::

      source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_6pe_dwc

   - For 6PE 350 Hz with MISC, you can select DPU IP via the following command.

   ::

      source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_6PE_misc

   - For 8PE 350 Hz, you can select DPU IP via the following command.

   ::

      source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_8pe_normal

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:

