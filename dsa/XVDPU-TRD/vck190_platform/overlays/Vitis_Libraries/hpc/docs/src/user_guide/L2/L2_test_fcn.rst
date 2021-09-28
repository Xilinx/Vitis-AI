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

.. _fcn_test_l2:

*******************************
FCN Kernel Test
*******************************

L2 FCN kernels have been tested against the implementation in C++ on host side.

Test FCN kernels
==============================
There are two pre-build L2 FCN kernels, one with 1 FCN CU (Compute Unit) and the other one with 4 FCN CUs. 
Both of the two kernels can be tested individually. 
To launch the testing process, please navigate to each testcase directory under **L2/tests/hw/mlp/**. 

.. code-block:: bash

  make run TARGET=sw_emu/hw_emu

The above command will test and verify the FCN kernel via Vitis software-emulation or hardware-emulation.
Once the emulations are pased, one can use the following command to build FPGA bitstream 
and launch the kernel on Alveo U250/U50 FPGA card. 

.. code-block:: bash

  make build TARGET=hw
  make run TARGET=hw
