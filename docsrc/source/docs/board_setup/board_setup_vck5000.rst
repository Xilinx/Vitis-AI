Setting up a Versal Accelerator Card
====================================

The Xilinx |reg| **DPUCVDX8H** for the Versal VCK5000 is a High Performance CNN processing engine.  The following instructions will help you to install the software and packages required to support the VCK5000.

As a first step, it is recommended that you select the appropriate DPU configuration for your application:

=== ================ =====================
No. Accelerator Card DPU IP
=== ================ =====================
1   VCK5000-PROD     DPUCVDX8H_4pe_miscdwc
2   VCK5000-PROD     DPUCVDX8H_6pe_dwc
3   VCK5000-PROD     DPUCVDX8H_6pe_misc
4   VCK5000-PROD     DPUCVDX8H_8pe_normal
=== ================ =====================

This selection depends on several factors:

- `misc` is selected if the models to be deployed leverage pooling and element-wise layers
- `dwc` is selected if the models to be deployed leverage Depthwise Convolution (eg. MobileNets)
- The number of `pe` or "processing engines" determines both performance and programmable logic resource utilization

If you are just getting started and are uncertain which to choose, you may wish to start with the `DPUCVDX8H_4pe_miscdwc` as it provides the most extensive operator support.  Please refer to `this user guide <https://docs.xilinx.com/r/en-US/pg403-dpucvdx8h/Configuration-Options>`__ for more extensive details regarding selection.


VCK5000-PROD Card Setup
-----------------------

A script is provided to drive the VCK5000-PROD card setup process.

.. note:: You should run this script on the host machine, OUTSIDE of the Docker container. After the script has executed successfully, manually reboot the host server once. For data center DPUs, Vitis AI 3.0 specifically leverages the 2022.2 versions of the Vitis tools, VCK5000 platform, XRT and XRM.

This script will detect the operating system of the host, and will download and install the appropriate packages for that operating system.  Please refer to :doc:`Host System Requirements <../reference/system_requirements>` prior to proceeding.

Execute this script as follows:

.. code-block::

   cd <Vitis-AI install path>/Vitis-AI/board_setup/vck5000
   source ./install.sh



The following installation steps are performed by this script:

1. XRT Installation. The `Xilinx RunTime (XRT) <https://github.com/Xilinx/XRT>`__ is a combination of userspace and kernel driver components supporting PCIe accelerator cards such as the VCK5000. 
2. XRM Installation. The `Xilinx Resource Manager (XRM) <https://github.com/Xilinx/XRM/>`__ manages and controls FPGA resources on the host. It is required by the runtime.
3. Installation of the VCK5000-PROD platform.
4. Installation of the DPU xclbin for the VCK5000-PROD platform.

After the script is executed successfully, use the XRT `xbutil` command to check that the installation was successful. The result should contain the correct information for System Configuration, XRT and Devices present.

.. code-block::

   /opt/xilinx/xrt/bin/xbutil examine

.. note:: Vitis AI 3.0 requires the use of a VCK5000-PROD card. Support for the pre-production VCK5000-ES1 card is not available in this release. If you do not have a production release card, you must use `Vitis AI 1.4.1 <https://github.com/Xilinx/Vitis-AI/tree/v1.4.1>`__.

Docker Container Environment Variable Setup
-------------------------------------------

First, ensure that you have cloned Vitis AI, entered the Vitis AI directory.  Start Docker. 

From inside the docker container, execute one of the following commands to set the required environment variables for the DPU.  Note that the chosen xclbin file must be in the ``/opt/xilinx/overlaybins`` directory prior to execution. There are four xclbin files to choose from.  Select the xclbin that matches your chosen DPU configuration.

- For the 4PE 350MHz configuration with pooling, elementwise and depthwise convolution support:

   .. code-block::
   
      source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_4pe_miscdwc
	  
- For the 6PE 350MHz configuration with depthwise convolution support:

   .. code-block::
   
      source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_6pe_dwc

- For the 6PE 350MHz configuration with pooling and elementwise support:

   .. code-block::
   
      source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_6PE_misc

- For the 8PE 350MHz base configuration:

   .. code-block::
   
      source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_8pe_normal


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
