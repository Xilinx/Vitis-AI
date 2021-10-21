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
   :keywords: BLAS, Library, Vitis BLAS Library, primitives, matrix storage
   :description: Vitis BLAS library L1 primitives are the C++ implementation of BLAS functions.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _user_guide_overview_content_l1:


Vitis BLAS L1 primitives are the C++ implementation of BLAS functions. These implementations are intended to be used by HLS (High Level Synthesis) users to build FPGA logic for their applications. 

1. Introduction
================
L1 primitives' implementations include computation and data mover modules. The computation modules always have stream interfaces. The data mover modules move data between vectors' and matrices' on-chip storage and the computation modules. This design strategy allows FPGA application programmers to quickly develop a high-performed logic by simply chaining serval computation and data mover modules together. The organization of Vitis BLAS L1 files and directories, as described below, reflects this design strategy.

* **L1/include/hw/xf_blas**: the directory that contains the computation modules
* **L1/include/hw/xf_blas.hpp**: the header file for L1 primitivers' users
* **L1/include/hw/helpers/dataMover**: the directory that contains the data mover modules
* **L1/include/hw/helpers/funcs**: the directory that contains the common computation modules used by several primitives
* **L1/include/hw/helpers/utils**: the directory that contains the utilities used in the primitives' implementations
* **L1/test/hw**: the directory that contains the top modules used for testing each implemented primitive, including its computation and data mover modules
* **L1/test/sw**: the directory that contains the testbench and test infrastructure support for the primitives
* **L1/test/build**: the directory that includes the vivado_hls script used for creating vivado_hls project to test each primitive's implementation
* **L1/test/run_test.py**: the python script for testing L1 primitives' implementations
* **L1/test/set_env.sh**: the shell script for setting up the environment used for testing L1 primitives.

More information about computation and data mover modules can be found in :doc:`L1 computation APIs<L1_compute_api>` and :doc:`L1 data mover APIs<L1_data_mover>`. 

2. L1 primitives' usage
========================
Vitis BLAS L1 primitives are intended to be used by hardware developers to implement an application or algorithm specific FPGA logic in HLS. The following example code shows a typical usage of L1 primitives.
 
The uut_top.cpp file in each primitive folder under L1/tests/hw directory provides a usage example of combining computation and data mover modules of the primitive. More information about testing L1 primitives can be found in :doc:`Test L1 primitives<L1_test>`.

3. Matrix storage used in L1 primitives 
========================================
The data mover components move matrices' and vectors' data stored in the on-chip memory, normally BRAM or URAM slices, into streams to feed the computation modules. The following matrix storage formats are supported.

* row-major matrix
* row-major symmetric matrix
* row-major packed symmetric matrix
* row-major triangular matrix
* row-major packed triangular matrix
* column-major banded matrix
* column-major banded symmetric matrix

More information about matrix storage formats and data mover components can be found in :doc:`Data movers used in L1 primitives<L1_data_mover>`.
 
