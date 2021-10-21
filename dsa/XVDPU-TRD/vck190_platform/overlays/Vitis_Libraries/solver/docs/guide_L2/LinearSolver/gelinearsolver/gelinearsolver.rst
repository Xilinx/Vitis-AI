
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
   :keywords: GELINEARSOLVER
   :description: This function solves a system of linear equation with general matrix along with multiple right-hand side vector.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


*******************************************************
General Linear Solver (GELINEARSOLVER)
*******************************************************

This function solves a system of linear equation with general matrix along with multiple right-hand side vector.

.. math::
      Ax=B

where :math:`A` is a dense general matrix of size :math:`m \times m`, :math:`x` is a vector need to be computed, and :math:`B` is input vector.
The maximum matrix size supported in FPGA is templated by NMAX.

