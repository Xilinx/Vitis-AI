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
   :keywords: Hash-Join, hashJoinMPU, combineCol, splitCol
   :description: Describes the structure and execution of Hash-Join Multi-Process-Unit version.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _guide-hash_join_mpu:

********************************************************
Internals of Hash-Join (Multi-Process-Unit Version)
********************************************************

.. toctree::
   :hidden:
   :maxdepth: 2

This document describes the structure and execution of Hash-Join Multi-Process-Unit Version,
implemented as :ref:`hashJoinMPU <cid-xf::database::hashJoinMPU>` function.

.. image:: /images/hash_join_mpu.png
   :alt: Hash Join MPU Structure
   :align: center

The Hash-Join primitive has a clustered design internally to utilize the advantage of high memory bandwidth in Xilinx FPGA.
Workload is distributed based on MSBs of hash value of join key to Processing Units (PU's), so that each PU can work independently.
Current design uses 8 PU's, served by 4 input channels through each of which a pair of key and payload can be passed in each cycle.

The number of PU is set to 8, as each PU requires a dedicated bank to avoid conflicts,
and due to DDR/HBM memory access delay, 4 channels can serve enough data to these PU's.
Each PU performs Hash-Join in 3 phases.

1. build bitmap: with small table as input, the number of keys falls into each hash values are counted.
   The number of counts are stored in bit vector in URAM.
   After a full scan of the small table, the bit vector is walked once,
   accumulating the counts to offsets of each hash.

2. build unit: the small table is read in again, and stored into DDR/HBM buffers of that PU.
   By referencing the bit vector in URAM created in previous phase,
   the kernel knows where to find empty slots for each key,
   and once a small table payload and key is written into DDR/URAM,
   the offset in bit vector is increased by 1,
   so that the next key of same hash value can be written into a different place.
   As the number of keys with each hash have been counted,
   such offset increase won't step into another key's slot.

3. probe unit: finally, the big table is read in, and again by referencing the bit vector with hash of key,
   we can know the offset of this hash and number of keys with the same hash.
   Then the possible matched key and payload pairs can be retrieved from DDR/HBM,
   and joined with big table payload after key comparison.

.. IMPORTANT::
   To reduce the storage size of hash-table on FPGA-board, the small table has to be scanned in TWICE,
   and followed by the big table ONCE.

.. CAUTION::
   Currently, this primitive expects unique key in small table.

This ``hashJoinMPU`` primitive has only one port for key input and one port for payload input.
If your tables are joined by multiple key columns or has multiple columns as payload,
please use :ref:`combineCol <cid-xf::database::combineCol>` to merge the column streams, and
use :ref:`splitCol <cid-xf::database::splitCol>` to split the output to columns.

There are two versions of this primitive currently, with different number of slots
for hash collision and key duplication. The version with more slots per hash entry has less
total row capacity, as summarized below:

  +--------------+----------------+
  | row capacity | hash slots     |
  +--------------+----------------+
  | 2M           | 262144 (0.25M) |
  +--------------+----------------+
  | 8M           | 512 (0.5K)     |
  +--------------+----------------+

