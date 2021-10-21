.. 
   Copyright 2019 - 2021 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _cg_test_l2:

*******************************
CG Kernel Test
*******************************

L2 CG kernels have been tested against the implementation in python. 
That is, a python based testing environment has been developed to generate random test inputs 
for each CG kernel, compute the golden reference, and finally compare the golden reference.
To run the testing process of L2 kernels, please follow the steps below.

Set up Python environment
=============================
Please follow the instructions described in :doc:`Python environment setup guide <../../pyenvguide>` 
to install anaconda3 and setup xf_hpc environment.
All testing should be run under xf_hpc environment.
Please deactivate xf_hpc environment after testing.

Set up Vitis environment
=================================
Please navigate to directory L2/tests, and change the setting of environment variable 
**TA_PATH** to point to the installation path of your Vitis 2021.1, and run following command to set up Vivado_hls environment.

.. code-block:: bash

   export XILINX_VITIS=${TA_PATH}/Vitis/2021.1
   export XILINX_VIVADO=${TA_PATH}/Vivado/2021.1
   source ${XILINX_VITIS}/settings64.sh

Test CG kernels
==============================
There are several pre-build L2 kernels and they can be tested individually. 
To launch the testing process, please navigate to each testcase directory under **L2/tests/cgSolver/**, 
and enter the following command for software emulation, hardware emulation or
running on hardware. 

.. code-block:: bash

  make run TARGET=sw_emu/hw_emu/hw


GEMV-based CG solver
=======================

.. code-block:: bash

  make run TARGET=hw_emu

The above command will test and verify forward kernel via Vitis hardware-emulation.
Once the emulations are pased, one can use the following command to build FPGA bitstream 
and launch the kernel on Alveo U280 FPGA or Alveo U50 FPGA. 

.. code-block:: bash

  make build TARGET=hw
  make run TARGET=hw

The paramters listed in the following table can be configured with **make** command.

+----------------+----------------+---------------------------------------+
|  Parameter     |  Default Value |  Notes                                |
+================+================+=======================================+
|  CG_numChannels|   1/8/16       |  No. parallel HBM channels for matrix |
+----------------+----------------+---------------------------------------+

SPMV-based CG solver
=======================
.. code-block:: bash

  make run TARGET=hw_emu

The above command will test and verify forward kernel via Vitis hardware-emulation.
Once the emulations are pased, one can use the following command to build FPGA bitstream 
and launch the kernel on Alveo U280 FPGA or Alveo U50 FPGA. 

.. code-block:: bash

  make build TARGET=hw
  make run TARGET=hw

