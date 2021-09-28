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
   :keywords: BLAS, Library, Vitis BLAS Library, data mover
   :description: Vitis BLAS library L1 data mover modules are used to move matrix and vector data between their on-chip storage and the input/output streams of the computation modules.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _user_guide_data_mover_l1:

***********************
L1 Data mover
***********************

The L1 data mover modules are used to move matrix and vector data between their on-chip storage and the input/output streams of the computation modules. These data movers are intended to be used in conjunction with computation modules to form the HLS implementations for BLAS level 1 and 2 functions. Users can find this usage in uut_top.cpp files of the BLAS function name folders under directory L1/tests.

1. Matrix storage format
=========================
The following matrix storage formats are supported by L1 data mover modules.

* row-based storage in a contiguous array
* packed storage for symmetric and triangular matrices
* banded storage for banded matrices

For symmetric, triangular and banded storage, both Up and Lo storage modes are supported. More details about each storage format can be found in `xf_blas/L1/matrix_storage`_.

.. _xf_blas/L1/matrix_storage: https://www.netlib.org/blas/blast-forum/chapter2.pdf

2. Data mover APIs
===================

.. toctree:
   :maxdepth: 2

.. include:: ./namespace_xf_linear_algebra_blas_DM.rst
   :start-after: Global Functions
