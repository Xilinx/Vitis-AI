
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
   :keywords: GEQRF
   :description: This function solves a system of linear equation with triangular coefficient matrix along with multiple right-hand side vector.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

*******************************************************
General QR Decomposition (GEQRF)
*******************************************************

This function computes QR factorization of matrix :math:`A`

.. math::
            A = Q R

where :math:`A` is a dense matrix of size :math:`m \times n`, :math:`Q` is a :math:`m \times n` matrix with orthonormal columns, and :math:`R` is an
upper triangular matrix.
The maximum matrix size supported in FPGA is templated by NRMAX and NCMAX.
