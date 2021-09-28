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

.. _rtm_test_l2:

*******************************
RTM Kernel Test
*******************************

L2 RTM kernels have been tested against the implementation in python. 
That is, a python based testing environment has been developed to generate random test inputs 
for each RTM kernel, compute the golden reference, and finally compare the golden reference.
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

Test RTM kernels
==============================
There are several pre-build L2 kernels and they can be tested individually. 
To launch the testing process, please navigate to each testcase directory under **L2/tests/hw**, 
and enter the following command for software emulation, hardware emulation or
running on hardware. 

.. code-block:: bash

  make run TARGET=sw_emu/hw_emu/hw


Test 2D RTM
=======================

Forward kernel
--------------------------------

.. code-block:: bash

  make run TARGET=sw_emu/hw_emu

The above command will test and verify forward kernel via Vitis software-emulation or hardware-emulation.
Once the emulations are pased, one can use the following command to build FPGA bitstream 
and launch the kernel on Alveo U280 FPGA. 

.. code-block:: bash

  make build TARGET=hw
  make run TARGET=hw

The paramters listed in the following table can be configured with **make** command.
Notice that **RTM_time** must be multiple of **RTM_numFSMs**.

.. table:: Parameters with make command 
    :align: center

    +----------------+----------------+------------------------------------+
    |  Parameter     |  Default Value |  Notes                             |
    +================+================+====================================+
    |  RTM_maxDim    |   1282         |  Compile time: One dimmention limit|
    +----------------+----------------+------------------------------------+
    |  RTM_NXB       |   40           |  Compile time: Boundary width      |
    +----------------+----------------+------------------------------------+
    |  RTM_NZB       |   40           |  Compile time: Boundary height     |
    +----------------+----------------+------------------------------------+
    |  NUM_numFSMs   |   2            |  Compile time: No.stream module    |
    +----------------+----------------+------------------------------------+
    |  RTM_nPE       |   4            |  Compile time: No.PE               |
    +----------------+----------------+------------------------------------+
    |  RTM_order     |   8            |  Compile time: Spatial Order       |
    +----------------+----------------+------------------------------------+
    |  RTM_height    |   10           |  Running time: Image total height  |
    +----------------+----------------+------------------------------------+
    |  RTM_width     |   10           |  Running time: Image total widht   |
    +----------------+----------------+------------------------------------+
    |  RTM_time      |   10           |  Running time: No.time             |
    +----------------+----------------+------------------------------------+

Backward kernel
--------------------------------

.. code-block:: bash

  make run TARGET=sw_emu/hw_emu

The above command will test and verify backward kernel via Vitis software-emulation or hardware-emulation.
Once the emulations are pased, one can use the following command to build FPGA bitstream 
and launch the kernel on Alveo U280 FPGA. 

.. code-block:: bash

  make run TARGET=hw

The paramters listed in the following table can be configured with **make** command.
Notice that **RTM_time** must be multiple of **RTM_numBSMs**.

.. table:: Parameters with make command 
    :align: center

    +----------------+----------------+------------------------------------+
    |  Parameter     |  Default Value |  Notes                             |
    +================+================+====================================+
    |  RTM_maxDim    |   1282         |  Compile time: One dimmention limit|
    +----------------+----------------+------------------------------------+
    |  RTM_NXB       |   40           |  Compile time: Boundary width      |
    +----------------+----------------+------------------------------------+
    |  RTM_NZB       |   40           |  Compile time: Boundary height     |
    +----------------+----------------+------------------------------------+
    |  NUM_numFSMs   |   2            |  Compile time: No.stream module    |
    +----------------+----------------+------------------------------------+
    |  RTM_nPE       |   4            |  Compile time: No.PE               |
    +----------------+----------------+------------------------------------+
    |  RTM_order     |   8            |  Compile time: Spatial Order       |
    +----------------+----------------+------------------------------------+
    |  RTM_height    |   10           |  Running time: Image total height  |
    +----------------+----------------+------------------------------------+
    |  RTM_width     |   10           |  Running time: Image total widht   |
    +----------------+----------------+------------------------------------+
    |  RTM_time      |   10           |  Running time: No.time             |
    +----------------+----------------+------------------------------------+

RTM kernel
--------------------------------

RTM kernel is a combination of forward kernel and backward kenrel. 
It fulfils the entire RTM algorithm.

.. code-block:: bash

  make run TARGET=sw_emu/hw_emu

The above command will test and verify RTM kernel via Vitis software-emulation or hardware-emulation.
Once the emulations are pased, one can use the following command to build FPGA bitstream 
and launch the kernel on Alveo U280 FPGA. 

.. code-block:: bash

  make run TARGET=hw

The paramters listed in the following table can be configured with **make** command.
Notice that **RTM_time** must be multiple of **RTM_numFSMs** and **RTM_numBSMs**.

.. table:: Parameters with make command 
    :align: center

    +----------------+----------------+------------------------------------+
    |  Parameter     |  Default Value |  Notes                             |
    +================+================+====================================+
    |  RTM_maxDim    |   1282         |  Compile time: One dimmention limit|
    +----------------+----------------+------------------------------------+
    |  RTM_NXB       |   40           |  Compile time: Boundary width      |
    +----------------+----------------+------------------------------------+
    |  RTM_NZB       |   40           |  Compile time: Boundary height     |
    +----------------+----------------+------------------------------------+
    |  NUM_numFSMs   |   4            |  Compile time: No.stream module    |
    +----------------+----------------+------------------------------------+
    |  NUM_numBSMs   |   4            |  Compile time: No.stream module    |
    +----------------+----------------+------------------------------------+
    |  RTM_nPE       |   2            |  Compile time: No.PE               |
    +----------------+----------------+------------------------------------+
    |  RTM_order     |   8            |  Compile time: Spatial Order       |
    +----------------+----------------+------------------------------------+
    |  RTM_height    |   10           |  Running time: Image total height  |
    +----------------+----------------+------------------------------------+
    |  RTM_width     |   10           |  Running time: Image total widht   |
    +----------------+----------------+------------------------------------+
    |  RTM_time      |   12           |  Running time: No.time             |
    +----------------+----------------+------------------------------------+

Test 3D RTM
===============

Forward kernel with HBC/RBC boundary condition
----------------------------------------------

.. code-block:: bash

  make run TARGET=sw_emu/hw_emu

The above command will test and verify forward kernel with HBC/RBC boundary condition via Vitis software-emulation or hardware-emulation.
Once the emulations are pased, one can use the following command to build FPGA bitstream 
and launch the kernel on Alveo U280 FPGA. 

.. code-block:: bash

  make build TARGET=hw
  make run TARGET=hw

The paramters listed in the following table can be configured with **make** command.
Notice that **RTM_time** must be multiple of **RTM_numFSMs**.
**RTM_z** must be less than **RTM_maxZZ** and be multiple of **RTM_nPEZ**.
**RTM_x** must be multiple of **RTM_nPEX**.


.. table:: Parameters with make command 
    :align: center

    +----------------+----------------+------------------------------------+
    |  Parameter     |  Default Value |  Notes                             |
    +================+================+====================================+
    |  RTM_maxY      |   280          |  Compile time: y-dimmention limit  |
    +----------------+----------------+------------------------------------+
    |  RTM_maxZ      |   180          |  Compile time: z-dimmention limit  |
    +----------------+----------------+------------------------------------+
    |  RTM_NXB       |   20           |  Compile time: Boundary width      |
    +----------------+----------------+------------------------------------+
    |  RTM_NYB       |   20           |  Compile time: Boundary width      |
    +----------------+----------------+------------------------------------+
    |  RTM_NZB       |   20           |  Compile time: Boundary height     |
    +----------------+----------------+------------------------------------+
    |  NUM_numFSMs   |   2            |  Compile time: No.stream module    |
    +----------------+----------------+------------------------------------+
    |  RTM_nPEX      |   4            |  Compile time: No.PE along X       |
    +----------------+----------------+------------------------------------+
    |  RTM_nPEZ      |   4            |  Compile time: No.PE along Z       |
    +----------------+----------------+------------------------------------+
    |  RTM_order     |   8            |  Compile time: Spatial Order       |
    +----------------+----------------+------------------------------------+
    |  RTM_x         |   10           |  Running time: Image x dim size    |
    +----------------+----------------+------------------------------------+
    |  RTM_y         |   10           |  Running time: Image y dim size    |
    +----------------+----------------+------------------------------------+
    |  RTM_z         |   10           |  Running time: Image z dim size    |
    +----------------+----------------+------------------------------------+
    |  RTM_time      |   10           |  Running time: No.time             |
    +----------------+----------------+------------------------------------+
