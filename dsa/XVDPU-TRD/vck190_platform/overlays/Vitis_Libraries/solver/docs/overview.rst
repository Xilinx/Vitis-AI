..
  copyright 2019 Xilinx, Inc.
  
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
   :keywords: Vitis, Solver, Library, Vitis Solver Library, overview, matrix, linear, eigenvalue
   :description: Vitis Solver Library provides a collection of matrix decomposition operations, linear solvers and eigenvalue solvers.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _overview:

.. toctree::
      :hidden:

Library Overview
=================

Vitis Solver Library provides a collection of matrix decomposition operations, linear solvers and eigenvalue solvers.
 
Currently this includes the following operations for dense matrix
 
* Matrix decomposition
   * Cholesky decomposition for symmetric positive definite matrix
   * LU decomposition without pivoting and with partial pivoting
   * QR decomposition for general matrix
   * SVD decomposition (single value decomposition) for symmetric matrix and non-symmetric matrix (Jacobi method)
 
* Linear solver
   * Tridiagonal linear solver (Parallel cyclic reduction method)
   * Linear solver for triangular matrix
   * Linear solver for symmetric and non-symmetric matrix
   * Matrix inverse for symmetric and non-symmetric matrix
 
* Eigenvalue solver
   * Jacobi eigenvalue solver for symmetric matrix

Requirements
------------

Software requirements
~~~~~~~~~~~~~~~~~~~~~
* Vitis™ Unified Software Platform |ProjectVersion|
* CentOS/RHEL 7.4, 7.5 and Ubuntu 16.04.4 LTS, 18.04.1 LTS.

Hardware requirements
~~~~~~~~~~~~~~~~~~~~~
* `Alveo U200 <https://www.xilinx.com/products/boards-and-kits/alveo/u200.html>`_
* `Alveo U250 <https://www.xilinx.com/products/boards-and-kits/alveo/u250.html>`_

License
-------

    Licensed using the `Apache 2.0 license <https://www.apache.org/licenses/LICENSE-2.0>`_::

    Copyright 2019-2020 Xilinx, Inc.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Trademark Notice
----------------

    Xilinx, the Xilinx logo, Artix, ISE, Kintex, Spartan, Virtex, Zynq, and
    other designated brands included herein are trademarks of Xilinx in the
    United States and other countries.  All other trademarks are the property
    of their respective owners.

