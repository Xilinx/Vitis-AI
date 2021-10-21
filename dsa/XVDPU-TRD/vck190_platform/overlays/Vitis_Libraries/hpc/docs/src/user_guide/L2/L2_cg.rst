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

.. _cg_kernels:

**************************
CG Kernels 
**************************

Conjugate Gradient solvers are implemented by multiple streaming kernels. 
In this repository, both sparse-matrix based and dense-matrix based solver kernel are provided. To
accelerate the convengence, a popular preconditioner, Jacobi preconditioner, is integrated with the
solver.


Usage and Benchmark
======================================

.. toctree::
   :maxdepth: 2

   benchmark/cg_gemv_jacobi.rst
   benchmark/cg_spmv_jacobi.rst

