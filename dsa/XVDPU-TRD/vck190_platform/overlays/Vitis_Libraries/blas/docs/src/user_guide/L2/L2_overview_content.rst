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
   :keywords: BLAS, Library, Vitis BLAS Library, L2 Kernel, Gemm
   :description: Vitis BLAS library L2 applications.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _user_guide_overview_content_l2:


Vitis BLAS L2 pre-defined kernels are the C++ implementation of BLAS functions. 
These implementations are intended to demonstrate how FPGA kernels are defined and how L1 primitive functions can be used by any Vitis users to build their kernels for theri applications. 

1. Introduction
================
L2 kernel implementations include memory datamovers and computation components composed by L1 primitive functions. 
The kernels always have memoy (DDR/HBM) interfaces. 
The data mover modules move data between vectors' and matrices' off-chip storage and the computation modules. 
The L1 primitive functions with stream interfaces can be quickly chained with the data mover modules together to form a computation kernel.
The organization of Vitis BLAS L2 files and directories, as described below, reflects this design strategy.

* **L2/include/hw/xf_blas/**: the directory that contains the kernel modules
* **L2/include/sw/**: the directory that contains the host modules
* **L2/test/hw**: the directory that contains the Makefiles used for testing each implemented kernel
   
More information about computation and data mover modules can be found in :doc:`L2 GEMM kernel<L2_gemm>`. 

2. L2 kernel usage
========================
Vitis BLAS L2 pre-defined kernels can be used in users' applications based on BLAS functions. These kernels are also examples to present how to use the L1 primitive funtions and datamovers to build a kernel.
