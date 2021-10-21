.. 
   Copyright 2019 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. meta::
   :keywords: BLAS, Library, Vitis BLAS Library, Vitis BLAS, level 3, test
   :description: Vitis BLAS level 3 provides test cases could build xclbin and run it.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _test_l3:

=====================
L3 API test
=====================
Vitis BLAS level 3 provides test cases could build xclbin and run it.

**1. Vitis BLAS L3 compilation**

All tests provided here could be built with compilation steps similar to the following, target could be either hw or hw_emu(for testing hw emulation)

.. code-block:: bash

  make host TARGET=hw
  
**2. Vitis BLAS L3 run**

Tests could be run with the following steps, target could be either hw or hw_emu(for testing hw emulation)

.. code-block:: bash

  make run TARGET=hw PLATFORM_REPO_PATHS=LOCAL_PLATFORM_PATH
