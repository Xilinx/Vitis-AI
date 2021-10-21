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
   :keywords: Vitis Sparse Matrix Library, primitive details
   :description: Vitis Sparse Matrix Library primitive implementation details.

.. _L1_dispCol:

**************************************************************************************
Column Vector Buffering and Distribution Implementation
**************************************************************************************

.. toctree::
   :maxdepth: 1

This page provides the column vector buffering and distribution implementation details. The following figure shows the column vector buffering and distribution logic. 

.. image:: /images/dispCol.png
   :alt: cscRow Diagram
   :align: center

- The input parameter streams contain the information of the size of each column vector block, the minimum and maximum column vector entry indices. 
- The ``dispColVec`` module reads the parameters and multiple column vector enties, buffers the column entires in its own on-chip memory and forward the rest parameters and vector entires to the next ``disColVec`` module. If the module is the last one in the chain, the forwarding logic is omitted. 
- After the buffereing operation, each ``dispColVec`` module reads out the data from the on-chip memory and sends them to the output stream to be processed by its own computation path. 
- Apart from buffering and reading data operations, the ``dispColVec`` module also aligns the data and pads the data according to ``SPARSE_parEntries`` and the minimumn and maximum row indices. 
