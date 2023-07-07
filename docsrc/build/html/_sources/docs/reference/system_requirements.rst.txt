Vitis AI Host (Developer) Machine Requirements
===============================================

The following table lists Vitis |trade| AI developer workstation system requirements:

+------------------------------------------------------------------------+--------------------------------------------------------------------------+
| Component                                                              | Requirement                                                              |
+========================================================================+==========================================================================+
| ROCm GPU (GPU is optional but strongly recommended for quantization)   | AMD ROCm GPUs supporting ROCm v5.4.1, requires Ubuntu 20.04              |
+------------------------------------------------------------------------+--------------------------------------------------------------------------+
| CUDA GPU (GPU is optional but strongly recommended for quantization)   | NVIDIA GPUs supporting CUDA 11.3 or higher, (eg: NVIDIA P100, V100, A100)|
+------------------------------------------------------------------------+--------------------------------------------------------------------------+
| CUDA Driver                                                            | NVIDIA-465.19.01 or higher for CUDA 11.3                                 |
+------------------------------------------------------------------------+--------------------------------------------------------------------------+
| Docker Version                                                         | 19.03 or higher, nvidia-docker2                                          |
+------------------------------------------------------------------------+--------------------------------------------------------------------------+
| Operating System                                                       | Ubuntu 20.04                                                             |
+------------------------------------------------------------------------+--------------------------------------------------------------------------+
|                                                                        | CentOS 7.8, 7.9, 8.1                                                     |
+------------------------------------------------------------------------+--------------------------------------------------------------------------+
|                                                                        | RHEL 8.3, 8.4                                                            |
+------------------------------------------------------------------------+--------------------------------------------------------------------------+
| CPU                                                                    | Intel i3/i5/i7/i9/Xeon 64-bit CPU                                        |
+------------------------------------------------------------------------+--------------------------------------------------------------------------+
|                                                                        | AMD EPYC 7F52 64-bit CPU                                                 |
+------------------------------------------------------------------------+--------------------------------------------------------------------------+

Vitis AI Supported Board Targets
---------------------------------

The following table lists target boards that are supported with pre-built board images by Vitis AI:

.. note:: Custom platform support for Edge/Embedded devices may be enabled by the developer through Vitis and Vivado |reg| workflows.

====================== ==================================
Family                 Supported Target
====================== ==================================
Alveo 		           V70 `Early Access <https://www.xilinx.com/member/v70.html#vitis_ai>`__ 
Zynq UltraScale+ MPSoC ZCU102 and ZCU104 Boards
Versal                 VCK190 and VCK5000 Boards
Versal AI Edge         VEK280 `Early Access <https://www.xilinx.com/member/vitis-ai-vek280.html>`__
Kria                   KV260
====================== ==================================

Alveo Card System Requirements
-------------------------------

Refer to the “System Requirements” section of the target Alveo card documentation.


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
