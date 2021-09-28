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

.. _L2_spmv_double_api:

************************************
Double Precision SPMV Kernel APIs
************************************


.. toctree::
      :maxdepth: 1

.. NOTE::
   The double precision SPMV implementation in the current release uses 16 HBM channels to store NNZ values and indices, 1 HBM channel to store input dense vector X, 2 HBM channels to store partition parameters and 1 HBM channel to store result Y vector.

.. include:: ../../rst/global.rst
      :start-after: _cid-assembleykernel:
