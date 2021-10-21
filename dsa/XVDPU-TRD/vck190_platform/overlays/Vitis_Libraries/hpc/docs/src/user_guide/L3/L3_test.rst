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

.. _test_l3:

****************************
L3 test
****************************

The L3 software APIs provided in the xf_hpc library is a set of application level APIs.
These APIs allow software developers to call C/C++ rutines for offloading computation 
tasks to the accelerators running on FPGA devives. Users do not need to have hardware
design experience to use these APIs. However, they do need to have proper Xilinx Runtime
(XRT) installed and a pre-build FPGA-based accelerator logic (.xclbin file) available.

Users can find XRT-based L3 APIs' implementations in the ``L3/include/sw`` directory.

Level 3 provides test case could build xclbin and run it. In order to generate data for test, python3 is needed as well. Please refer to :doc:`Python environment setup guide<../../pyenvguide>` for more info.

**1. L3 test compilation**

All tests provided here could be built with compilation steps similar to the following

.. code-block:: bash

  make host TARGET=hw
  
**2. L3 xclbin generation**

.. code-block:: bash

  make build TARGET=hw PLATFORM_REPO_PATHS=LOCAL_PLATFORM_PATH
  
**3. L3 test run**

Tests could be run with the following steps

.. code-block:: bash

  make run TARGET=hw PLATFORM_REPO_PATHS=LOCAL_PLATFORM_PATH


