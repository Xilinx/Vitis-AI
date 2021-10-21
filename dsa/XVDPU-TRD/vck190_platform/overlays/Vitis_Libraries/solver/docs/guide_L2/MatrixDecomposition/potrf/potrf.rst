
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
   :keywords: POTRF, Decomposition, Cholesky, SPD, matrix
   :description: This function computes the Cholesky decomposition of matrix.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

**********************************************
Cholesky Decomposition for SPD matrix (POTRF)
**********************************************

This function computes the Cholesky decomposition of matrix :math:`A`

.. math::
    A = L {L}^T

where :math:`A` is a dense symmetric positive-definite matrix of size :math:`m \times m`, :math:`L` is a lower triangular matrix, and :math:`{L}^T` is the transposed matrix of :math:`L`.
The maximum matrix size supported in FPGA is templated by NMAX.
