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
   :keywords: dynamic-evaluation, dynamicEval
   :description: Describes the structure and execution of the dynamic evaluation module.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _guide-dynamic_eval:

********************************************************
Internals of Dynamic-Evaluation
********************************************************

.. toctree::
   :hidden:
   :maxdepth: 1

This document describes the structure and execution of dynamic evaluation module,
implemented as :ref:`dynamicEval <cid-xf::database::dynamicEval>` function.

The structure of ``dynamicEval`` is described as below. The primitive has 4-stream and 4-constant as inputs. 
The data type of input stream and constant is a template parameter in API, and the maximum width of constant is limited to 64-bit.

.. image:: /images/dynamic_eval_top.png
   :alt: Dynamic Evaluation Top Structure
   :align: center

The primitive is a tree-shaped structure with evaluation cells in each node. 
Each evaluation cell provides 4 kinds of computing unit, which are: compare, boolean algebra, multiplex and math compute.
For comparator and boolean algebra, the results are in boolean type, while multiplex and math compute will generate non-boolean results.
The internal of dynamic eval will expread two types of results from one cell to the next level of cells.
There are two types of cell design. For Cell1-Cell4, as the level-1 cells, the two inputs which are stream and constant respectively. 
For Cell5-Cell7, as the internal cells, which need process both boolean and non-boolean results from previous level. Cell1-Cell4 is shown as:

.. image:: /images/dynamic_eval_l1_cell.png
   :alt: Dynamic Evaluation Cell 1-4 Structure
   :align: center

While Cell5-Cell7 has more inputs to select:

.. image:: /images/dynamic_eval_l2_cell.png
   :alt: Dynamic Evaluation Cell 5-7 Structure
   :align: center

The configuration of the primitive is defined as below, and the bits
are concatenated without padding from top to bottom in LSB to MSB order.

+----------+--------------+--------+
| Type     | Usage        | Size   |
+==========+==============+========+
| Operator | Output Mux   | 1 bit  |
|          +--------------+--------+
|          | Strm Empty   | 4 bit  |
|          +--------------+--------+
|          | Cell1 OP     | 4 bit  |
|          +--------------+--------+
|          | Cell2 OP     | 4 bit  |
|          +--------------+--------+
|          | Cell3 OP     | 4 bit  |
|          +--------------+--------+
|          | Cell4 OP     | 4 bit  |
|          +--------------+--------+
|          | Cell5 OP     | 4 bit  |
|          +--------------+--------+
|          | Cell6 OP     | 4 bit  |
|          +--------------+--------+
|          | Cell7 OP     | 4 bit  |
+----------+--------------+--------+
| Operand  | C1           | 64 bit |
|          +--------------+--------+
|          | C2           | 64 bit |
|          +--------------+--------+
|          | C3           | 64 bit |
|          +--------------+--------+
|          | C4           | 64 bit |
+----------+--------------+--------+

To automatically generate its configuration, please refer to the test case in ``L3/tests/sw/dynamic_alu_host/test.cpp``
To manually generate configuration, please refer to built-in docs in ``L1/include/hw/xf_database/dynamic_eval.hpp``
