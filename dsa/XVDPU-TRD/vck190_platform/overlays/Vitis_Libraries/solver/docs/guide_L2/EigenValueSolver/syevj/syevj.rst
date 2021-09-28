
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
   :keywords: SYEVJ, Eigenvalue, Solver, Jacobi, Eigen
   :description: Symmetric Matrix Jacobi based Eigen Value Decomposition (SYEVJ).
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


*******************************************************
Eigenvalue Solver (SYEVJ)
*******************************************************

Symmetric Matrix Jacobi based Eigen Value Decomposition (SYEVJ)

.. math::
  A U = U \Sigma

where :math:`A` is a dense symmetric matrix of size :math:`m \times m`, :math:`U` is a :math:`m \times m` matrix with orthonormal columns, each column of U is the eigenvector :math:`v_{i}`, and :math:`\Sigma` is diagonal matrix, which contains the eigenvalues :math:`\lambda_{i}` of matrix A.
The maximum matrix size supported in FPGA is templated by NMAX.
