
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
   :keywords: GETRF_NOPIVOT, Decomposition
   :description: This function computes the LU decomposition (without pivoting) of matrix.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

*******************************************************
Lower-Upper Decomposition (GETRF_NOPIVOT)
*******************************************************

This function computes the LU decomposition (without pivoting) of matrix :math:`A`

.. math::
    A = L U

where :math:`A` is a dense matrix of size :math:`m \times m`, :math:`L` is a lower triangular matrix with unit diagonal, and :math:`U` is a upper triangular matrix. This function does not implement pivoting.
The maximum matrix size supported in FPGA is templated by NMAX.
