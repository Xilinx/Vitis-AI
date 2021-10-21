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

.. _user_guide_test_l1:

*******************************
L1 Test
*******************************

All L1 primitives' implementations have been tested against implementations in python. 
That is, a python based testing environment has been developed to generate random test inputs 
for each primitive, compute the golden reference, and finally compare the golden reference 
with the csim and cosim outputs of the primitive to verify the correctness of the implementation.
To run the testing process of L1 primitives, please follow the steps below.

1. Set up Python environment
=============================
Please follow the instructions described in :doc:`Python environment setup guide <../../pyenvguide>` 
to install anaconda3 and setup xf_hpc environment.
All testing should be run under xf_hpc environment.
Please deactivate xf_hpc environment after testing.

2. Set up Vitis_hls environment
=================================
Please navigate to directory L1/tests, and change the setting of environment variable 
**TA_PATH** to point to the installation path of your Vitis 2021.1, 
and run following command to set up Vivado_hls environment.

.. code-block:: bash

   export XILINX_VITIS=${TA_PATH}/Vitis/2021.1
   source ${XILINX_VITIS}/settings64.sh

3. Test L1 primitives
==============================
To launch the testing process, please navigate to the directory **L1/tests/hw/**.
There are three functions under testing in this direcotry. For each function,
there are several test cases with various configurations under **./tests/** directory. 
For each test case, please use following commands to check the Makefile usage

.. code-block:: bash

    make help

Makefile usage example:

.. code-block:: bash

    make run CSIM=1 CSYNTH=1 COSIM=1 DEVICE=<FPGA platform> PLATFORM_REPO_PATHS=<path to platform directories>

Command to run the selected tasks for specified device. Valid tasks are 'CSIM', 'CSYNTH', 'COSIM', 'VIVADO_SYN', 'VIVADO_IMPL'. 

'PLATFORM_REPO_PATHS' variable is used to specify the paths in which the platform files will be searched for.

'DEVICE' is case-insensitive and support awk regex. For example:

.. code-block:: bash

    make run DEVICE='u280.*xdma' COSIM=1

It can also be an absolute path to a platform file. 
