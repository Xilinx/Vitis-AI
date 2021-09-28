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
   :keywords: Hash-Anti-Join,  hashAntiJoin, combineCol, splitCol
   :description: Describes the structure and execution of Hash-Anti-Join.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



.. _guide-hash_anti_join:

********************************************************
Internals of Hash-Anti-Join
********************************************************

.. toctree::
   :hidden:
   :maxdepth: 2

This document describes the structure and execution of Hash-Anti-Join, implemented as :ref:`hashAntiJoin <cid-xf::database::hashAntiJoin>` function.
Hash-Anti-Join is a special Join algorithm which returns rows from the first table which do not exist in the second table. It is based on the framework of ``Hash-Join-v3`` as shown in the picture below:

.. image:: /images/hashJoinV3Structure.png
   :alt: Hash Join MPU Structure
   :align: center

The number of PU is set to 8, as each PU requires two dedicated bank to temporarily store rows in small table (one for base rows 
and another for overflow rows). Due to DDR/HBM memory access delay, 4 channels can serve enough data to these PUs.
Each PU performs Hash-Join in 3 phases, and the detail is described in :ref:`guide-hash_join_v3`.  

The difference between Hash-Join and Hash-Anti-Join is only about the Join module, so that most of functions in Hash-Anti-Join are the same as Hash-Join. 
The Join function is replaced with Anti-Join function in Hash-Anti-Join primitive.

.. IMPORTANT::
   Make sure the size of small table is smaller than the size of HBM buffers. Small table and big table should be fed only ONCE.

.. CAUTION::
   Currently, this primitive expects unique key in small table.

The primitive has only one port for key input and one port for payload input.
If your tables are joined by multiple key columns or has multiple columns as payload,
please use :ref:`combineCol <cid-xf::database::combineCol>` to merge the column streams, and
use :ref:`splitCol <cid-xf::database::splitCol>` to split the output to columns.

There is a deep relation in the template parameters of the primitive. 
In general, the maximum capacity of rows and depth of hash entry is limited by the size of HTB. 
Each PU has one HTB in this design, and the size of one HTB is equal to the size of one pseudo-channel in HBM. 
Here is an example of row capacity when PU=8:

  +----------+----------+--------------+--------------+--------------+--------------------------+-------------------+
  | HTB Size | Key Size | Payload Size | Row Capacity |  Hash Entry  | Max Depth for Hash Entry | Overflow Capacity |
  +----------+----------+--------------+--------------+--------------+--------------------------+-------------------+
  | 8x256MB  |  32 bit  |   32 bit     |      64M     |     1M       | 63 (base rows take 63M)  |         1M        |
  +----------+----------+--------------+--------------+--------------+--------------------------+-------------------+
  | 8x256MB  | 128 bit  |   128 bit    |      16M     |     1M       | 15 (base rows take 15M)  |         1M        |
  +----------+----------+--------------+--------------+--------------+--------------------------+-------------------+

The Number of hash entry is limited by the number of URAM in a single SLR. For example, there are 320 URAMs in a SLR of U280, and 1M hash entry will take 192 URAMs (96 URAMs for base hash counter + 96 URAMs for overflow hash counter). Because the number of hash entry must be the power of 2, 1M hash entry is the maximum for U280 to avoid crossing SLR logic which will lead to bad timing performance of the design.

