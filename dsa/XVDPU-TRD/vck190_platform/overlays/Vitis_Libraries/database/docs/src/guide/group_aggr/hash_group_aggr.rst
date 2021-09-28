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
   :keywords: Hash-Group-Aggregate, hashGroupAggregate
   :description: Describes the structure and execution of the Hash-Group-Aggregate (Generic Version).
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _guide-hash_aggr_general:

********************************************************
Internals of Hash-Group-Aggregate (Generic Version)
********************************************************

.. toctree::
   :hidden:
   :maxdepth: 2

This document describes the structure and execution of Hash-Aggr General Version,
implemented as :ref:`hashGroupAggregate <cid-xf::database::hashGroupAggregate>` function.

.. image:: /images/hashGroupAggregate.png
   :alt: Hash aggregate general module Structure
   :align: center

The Hash-Group-Aggregate (Generic Version) primitive is a primitive to accelerate Group-by algorithm.
It is designed to handle group aggregate with the capability of computing massive data within the utilization of high memory bandwidth.

This primitive can accept multiple input rows for key and payload on each cycle because of its multiple Processing Units (PUs).
The processing of multi-channel and multi-column is provided to work with scan module so that it could match the speed of DDR/HBM.
The input key and payload will be updated and stored in URAM. If hash overflow occurs, overflow key and payload will be sent to HBM which are exclusive for each PU.
When the kernel has finished one flow and there are still left keys which are not aggregated, the kernel will automatically scan the HBM again for the left keys and aggregate until all keys are aggregated.
A single PU will connect two HBM ports for a ping-pong process during different work round. 
The max distinct of key aggregated in one work flow is stable, which is 2^(_WHashHight + _WHashLow). The maximum hash overflow should not exceed the storage size of the HBM.
The data should be aligned to little-end.

There are 6 runtime programmable functions for calculating the payload, which are: 

- xf::database::AOP_MIN 
- xf::database::AOP_MAX 
- xf::database::AOP_SUM
- xf::database::AOP_MEAN
- xf::database::AOP_COUNT
- xf::database::AOP_COUNTNONZERO

The details of API description is shown in :ref:`hashGroupAggregate <cid-xf::database::hashGroupAggregate>`. A minimal setup for ``hashGroupAggregate`` should be:

- WKey=8
- KeyNM=1
- WPay=8
- PayNM=1
- HashMode=1
- WHashHigh=1
- WHashLow=12
- CHNM=1
- WBuffer=32
- BurstLenW=32
- BurstLenR=32

A maximal setup could be:

- WKey=64
- KeyNM=8
- WPay=64
- PayNM=8
- HashMode=1
- WHashHigh=2
- WHashLow=16
- CHNM=4
- WBuffer=512
- BurstLenW=32
- BurstLenR=32

A recommended setup (tested in GQE kernel) should be:

- WKey=32
- KeyNM=8
- WPay=32
- PayNM=8
- HashMode=1
- WHashHigh=2
- WHashLow=17
- CHNM=4
- WBuffer=512
- BurstLenW=32
- BurstLenR=32

For typical setup, the utilization in Vivado report on U280 card is:

  +------+------+--------+----------+--------+-----------+
  | URAM | BRAM |   LUT  | Register |  Freq  | HBM ports |
  +------+------+--------+----------+--------+-----------+
  |  256 |  46  | 140000 |  190000  |  230M  |    8      |
  +------+------+--------+----------+--------+-----------+

In the recommended setup, the primitive can process maximum of 4(channel) x 8(column) group key and 4(channel) x 8(column) payload per cycle, and the width of key and payload is 32. 
It has 4 PU and takes 8 AXI ports as ping-pong buffer for processing hash overflow. It can perform <sum/average/min/max/count/count_non_zero> which controlled by runtime parameter. 
For sum and average, the output have 64-bit (split into 32 high bit and 32 low bit) to avoid overflow in computing. 

This ``hashGroupAggregate`` primitive have multiple columns of input and output port for key and payload.

