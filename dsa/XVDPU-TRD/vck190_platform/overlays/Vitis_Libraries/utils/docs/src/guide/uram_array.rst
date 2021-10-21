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

.. _guide-class-uram_array:

********************************
Internals of UramArray
********************************

.. toctree::
   :hidden:
   :maxdepth: 2

The :ref:`UramArray <cid-xf::common::utils_hw::UramArray>` class aims to
help users to achive faster update rate to data stored in URAM blocks.

Work Flow
=========

.. figure:: /images/uram_array.png
    :alt: bit field
    :align: center

This module enables fast data update by creating a small history cache of
recently written data in register beside the URAM blocks.
Upon data read, it will lookup the address in recent writes, and forward
the result if match is found.

It also provides a handy interface for initializing multiple URAM blocks
used as an array in parallel.

Storage Layout
==============

URAM blocks have fixed width of 72-bit, so our storage layout depends on how
wide each element is.

When the data element has no more than 72bits, the helper class will try to
store as many as possible within 72bits and pad zeros when space is left.
For example, to store 20k 16-bit elements, 2 URAMs would be used,
as each line can store 4 elements and each URAM has fixed depth of 4k.

When the data element has more than 72bits, the helper class will use line of
multiple URAM blocks to store each element. This ensures that each cycle
we can initiate an element access.
So to store 10k 128-bit elements, 6 URAM blocks are required.


Resources
=========

The hardware resources for 10k elements in post-Vivado report are listed in
table below:

.. table:: Hardware resources for URAM
    :align: center

    +-------------+----------+----------+-----------+-----------+-------------+
    |    _WData   |   URAM   |    FF    |    LUT    |  Latency  |  clock(ns)  |
    +-------------+----------+----------+-----------+-----------+-------------+
    |      64     |     3    |   3215   |   1868    |   10243   |    2.046    |
    +-------------+----------+----------+-----------+-----------+-------------+
    |      128    |     6    |   4000   |   2457    |   10243   |    2.046    |
    +-------------+----------+----------+-----------+-----------+-------------+

