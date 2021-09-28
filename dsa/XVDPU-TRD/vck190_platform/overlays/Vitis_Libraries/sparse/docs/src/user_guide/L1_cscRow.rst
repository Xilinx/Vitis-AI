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

.. _L1_cscRow:

**************************************************************************************
Row-wise Accumulator Implementation
**************************************************************************************

.. toctree::
   :maxdepth: 1
This page provides the row-wise accumulator implementation details. The following figure shows the row-wise accumulator logic. 

.. image:: /images/cscRow.png
   :alt: cscRow Diagram
   :align: center

- There are three input streams, namely the column vector value stream, the NNZ value stream and the NNZ row indices stream. 
- Each stream contains multiple entries being processed in parallel. For example, there are 4 entries being processed in parallel as shown in the figure. 
- The number of parallel entries can be configured at the compile time by ``SPARSE_parEntries``. 
- The ``xBarRow`` logic, illustrated by the ``formRowEntry``, ``split`` and ``merge`` modules in the figure, carries out the multiplication of the column value and the NNZ value, and distributes the results and their corresponding row indices to the accumulator logic implemented by the ``rowMemAcc`` logic. 
- There are multiple on-chip memory blocks in each ``rowMemAcc`` logic to store the accumulation results and remove the floating point accumulation bubbles. 
- Once the accumulation operation is finished, each ``rowMemAcc`` module will read out the data in the on-chip memory and send them to ``rowAgg`` module to be merged together and written to the device memory, for example, HBM.
