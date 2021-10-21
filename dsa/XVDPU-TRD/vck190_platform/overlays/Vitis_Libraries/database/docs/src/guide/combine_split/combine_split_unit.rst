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
   :keywords: combine-split-unit, combineCol, splitCol
   :description: Describes the structure and execution of the Combine-Split-Unit.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _guide-combine_split_unit:

********************************************************
Internals of Combine-Split-Unit
********************************************************

.. toctree::
   :hidden:
   :maxdepth: 1 

This document describes the structure and execution of Combine-Split-Unit,
implemented as :ref:`combineCol <cid-xf::database::combineCol>` function and :ref:`splitCol <cid-xf::database::splitCol>` function.

.. image:: /images/combine_unit.png
   :alt: Combine Unit Structure
   :align: center

.. image:: /images/split_unit.png
   :alt: Split Unit Structure
   :align: center

The Combine Unit primitive is used to combine two or more streams into one wider stream. And the Split Unit is used to split one big stream into several thinner streams. 
Due to different numbers of input streams of combineUnit / output streams of spiltUnit. Four versions of combine/split unit are provided, including:

- 2-stream-input combine unit

- 3-stream-input combine unit

- 4-stream-input combine unit

- 5-stream-input combine unit

- 2-stream-output split unit

- 3-stream-output split unit

- 4-stream-output split unit

- 5-stream-output split unit

For the combine unit, the input streams are combined from left to right, with the corresponding inputs from stream1 to streamN. (aka. output stream = [input stream1, input stream2, ..., input streamN]).

For the split unit, the output streams are split from right to left, with the corresponding inputs from stream1 to streamN. (aka. [output streamN, ..., output stream2, output stream1] = input stream). 

.. CAUTION::
    - All input/output streams are ap_uint<> data type.
    - The maximum number of supported streams are 5 for both combine and split unit. When the input/output stream numbers are more than 5, the combination of 2 or more combine/split unit are required.

