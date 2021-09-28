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
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _L1_xBarCol:

**************************************************************************************
Scatter-Gather Logic Implementation
**************************************************************************************

.. toctree::
   :maxdepth: 1
This page provides the implementation details of the scatter-gather logic for selecting column vector entries.

The following figure shows the scatter-gather logic: 

.. image:: /images/xBarCol.png
   :alt: xBarCol Diagram
   :align: center

- The input column vector and column pointer data streams contain multiple entries, for example, 4 entries as shown in the diagram. 
- The number of entries in the stream can be configured at compile time by ``SPARSE_parEntries``. 
- Split logic distributes the column vector values into different single-entry streams according to their corresponding column pointer values. 
- The merge logic looks through all the streams and merges multiple single entry streams into one stream with multiple column vector values. 
