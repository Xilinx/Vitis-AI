.. 
   Copyright 2019 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2021.1 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2021.1
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. meta::
   :keywords: SPARSE, Library, Vitis SPARSE Library, linear algebra, Subroutines
   :description: Vitis SPARSE library release notes.

.. _release_note:

Release Note
============

.. toctree::
   :hidden:
   :maxdepth: 1

2020.1
----

The 2020.1 release introduces HLS primitives for SPARSE (Basic Linear Algebra Subroutines) operations. 
These primitives are implemented with ``HLS::stream`` interfaces to allow them to operate in parallel
with other hardware components. 

2021.1
----

The 2021.1 release introduces double precision SpMV (Sparse Matrix dense Vector multiplication) implementation
with L2 kernels. 
